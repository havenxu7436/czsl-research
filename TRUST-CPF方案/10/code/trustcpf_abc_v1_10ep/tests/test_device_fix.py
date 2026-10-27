import copy,hashlib,importlib.util,json,shutil,tempfile,unittest
from pathlib import Path
import torch
from test_competition import fixture,tiny_config
from trustcpf import competition_workflow as cw,competition_training as ct,matched_train as train
from trustcpf.common import ROOT,read_json,write_json,load_torch,sha

spec=importlib.util.spec_from_file_location('device_fix',ROOT/'scripts/fix_device_check.py')
fix=importlib.util.module_from_spec(spec);spec.loader.exec_module(fix)

class DeviceFixTests(unittest.TestCase):
    def test_migration_preserves_checkpoints_and_other_identity_fields(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);work=root/'work';(root/'trustcpf').mkdir()
            new=(ROOT/fix.KEY).read_bytes();old=new.replace(fix.NEW.encode(),fix.OLD.encode())
            self.assertEqual(fix.sha(old),fix.OLD_SHA)
            (root/fix.KEY).write_bytes(old);(root/'run.py').write_bytes(b'pass\n')
            identity=dict(config={'epochs':10},source_files={'x':'unchanged'},implementation={fix.KEY:fix.OLD_SHA,'run.py':sha(root/'run.py')})
            write_json(work/'progress/global/identity.json',identity);write_json(root/'MANIFEST.json',identity['implementation'])
            cp=work/'seed3351/teachers/1/last.pt';cp.parent.mkdir(parents=True);cp.write_bytes(b'existing epoch10 checkpoint')
            before=cp.read_bytes();fix.repair(root,work);fix.repair(root,work)
            self.assertEqual(cp.read_bytes(),before);self.assertEqual(sha(root/fix.KEY),fix.NEW_SHA)
            expected=copy.deepcopy(identity);expected['implementation'][fix.KEY]=fix.NEW_SHA
            self.assertEqual(read_json(work/'progress/global/identity.json'),expected)
            self.assertEqual(read_json(work/'progress/global/identity.json.before_device_fix'),identity)
            (root/'run.py').write_bytes(b'changed\n')
            with self.assertRaises(ValueError):fix.repair(root,work)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required for the regression')
    def test_cuda_ten_epochs_resume_and_completed_checkpoint(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);cfg=tiny_config();cfg.update(stop_images_per_group=2,risk_images_per_group=1,stage_epochs=10)
            a=fixture(root,cfg);a.work=str(root/'prior');cw.prepare(a);ct.teacher(a,1)
            w=root/'next';w.mkdir()
            for n in ('recipe.json','inputs/bank.pt'):
                (w/n).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(Path(a.work)/n,w/n)
            p=w/'old/teachers/1/last.pt';p.parent.mkdir(parents=True)
            shutil.copy2(Path(a.work)/'teachers/1/last.pt',p)
            write_json(w/'splits/fold1.json',train.add_stopping_set(read_json(Path(a.work)/'splits/fold1.json'),cfg))
            a.work=str(w);a.device='cuda';fresh=root/'fresh';shutil.copytree(w,fresh)
            train.train_teacher(a,1,until_epoch=3);train.train_teacher(a,1)
            b=copy.copy(a);b.work=str(fresh);train.train_teacher(b,1)
            resumed=load_torch(w/'teachers/1/last.pt');direct=load_torch(fresh/'teachers/1/last.pt')
            self.assertEqual(resumed['epoch'],10);self.assertEqual(len(resumed['history']),11)
            self.assertEqual(resumed['best_epoch'],direct['best_epoch'])
            for k,v in direct['model'].items():torch.testing.assert_close(v,resumed['model'][k],rtol=0,atol=0)
            before=sha(w/'teachers/1/last.pt');(w/'teachers/1/contract.json').unlink()
            train.train_teacher(a,1)
            self.assertEqual(sha(w/'teachers/1/last.pt'),before)
            contract=read_json(w/'teachers/1/contract.json');self.assertEqual(contract['epochs_completed'],10)
            write_json(ROOT/'validation/device_fix_gpu.json',dict(passed=True,gpu=torch.cuda.get_device_name(0),
                torch=torch.__version__,epochs=10,epoch3_resume_bitwise_equal=True,completed_checkpoint_no_retraining=True))

if __name__=='__main__':unittest.main()
