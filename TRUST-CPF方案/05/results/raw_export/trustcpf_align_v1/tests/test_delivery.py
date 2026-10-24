import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from trustcpf.common import ROOT, sha


class DeliveryTests(unittest.TestCase):
    def test_pack_manifest_hash_and_utf8_extraction_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'trustcpf_align_v1';scripts=root/'scripts';scripts.mkdir(parents=True)
            work=root/'work';work.mkdir()
            for name in ('pack_results.py','extract_results.py'):
                shutil.copy2(ROOT/'scripts'/name,scripts/name)
            (root/'说明.md').write_text('结果与公式\n',encoding='utf8')
            (work/'status.json').write_text(json.dumps(dict(complete=True)),encoding='utf8')
            (work/'array.bin').write_bytes(bytes(range(256))*100)
            env=dict(os.environ,CPF_WORK=str(work),PYTHONIOENCODING='utf-8')
            subprocess.run([sys.executable,str(scripts/'pack_results.py')],env=env,check=True,capture_output=True)
            desc=json.loads((root/'download.json').read_text(encoding='utf8'));archive=Path(desc['archive'])
            self.assertEqual(sha(archive),desc['sha256']);self.assertEqual(archive.stat().st_size,desc['bytes'])
            checksum=Path(str(archive)+'.sha256').read_bytes();self.assertNotIn(b'\r',checksum)
            self.assertNotIn(b'\xef\xbb\xbf',checksum)
            dest=Path(td)/'restored'
            subprocess.run([sys.executable,str(scripts/'extract_results.py'),str(archive),str(dest)],env=env,check=True,capture_output=True)
            self.assertEqual(sha(root/'说明.md'),sha(dest/'trustcpf_align_v1/说明.md'))
            self.assertEqual(sha(work/'array.bin'),sha(dest/'trustcpf_align_v1/work/array.bin'))

    def test_extractor_rejects_parent_path(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);archive=root/'escape.tar.gz'
            with tarfile.open(archive,'w:gz') as t:
                member=tarfile.TarInfo('../outside.txt');member.size=1;t.addfile(member,io.BytesIO(b'x'))
            run=subprocess.run([sys.executable,str(ROOT/'scripts/extract_results.py'),str(archive),str(root/'extract')],capture_output=True)
            self.assertNotEqual(run.returncode,0);self.assertFalse((root/'outside.txt').exists())


if __name__=='__main__':unittest.main()
