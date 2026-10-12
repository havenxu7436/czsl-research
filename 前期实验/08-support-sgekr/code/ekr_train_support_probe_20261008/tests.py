"""Meaningful CPU checks for the TRAIN-only diagnostic mathematics."""
import unittest
import numpy as np
from probe import analyze, evidence_vectors, stats, unit

def fixture():
    names=np.eye(300)[:16]
    # Each attribute occurs in two objects and each pair has six images.
    a=np.repeat(np.arange(16),12);o=np.tile(np.repeat([0,1],6),16)
    pair=np.repeat(np.arange(32),6)
    x=names[a].copy()
    ids=['image_%03d'%i for i in range(len(a))]
    slots=np.repeat(names[:,None,:],4,axis=1)
    shuffle=slots.copy();shuffle[:,1:]=slots[np.roll(np.arange(16),1),1:]
    b=dict(unit=names,units=slots,mask=np.ones((16,4),bool),shuffle_units=shuffle,shuffle_mask=np.ones((16,4),bool))
    return x,dict(attr=a,obj=o,pair=pair),ids,b,['a%d'%i for i in range(16)],['o0','o1']

class Tests(unittest.TestCase):
    def test_missing_slots_do_not_dilute_evidence(self):
        u=np.eye(4)[:,None,:].repeat(4,axis=1)
        m=np.ones((4,4),bool);m[:,2:]=False
        np.testing.assert_allclose(evidence_vectors(u,m),np.eye(4)/2)

    def test_margin_excludes_gold_and_unavailable_rivals(self):
        m,c=stats(np.array([[3.,2.,100.]]),np.array([0]),np.array([[True,True,False]]))
        self.assertEqual(m[0],1);self.assertTrue(c[0])

    def test_folds_are_disjoint_and_same_object_rivals_apply(self):
        values=fixture();r=analyze(*values)
        folds=np.array(r['folds'])
        for pi in np.unique(values[1]['pair']):
            self.assertEqual((folds[values[1]['pair']==pi]==0).sum(),3)
        self.assertEqual(len(r['crossfit']),4)
        self.assertFalse(r['decision']['official_EKR_confirmation_allowed'])
        self.assertEqual(r['decision']['new_optimizer_steps'],0)

    def test_own_name_shuffle_corruption_is_rejected(self):
        values=fixture();values[3]['shuffle_units'][:,0]=0
        with self.assertRaises(ValueError):analyze(*values)

    def test_wrong_pair_image_count_is_rejected(self):
        x,lab,ids,b,attrs,objs=fixture()
        lab={k:v[:-1] for k,v in lab.items()}
        with self.assertRaises(ValueError):analyze(x[:-1],lab,ids[:-1],b,attrs,objs)

    def test_macro_average_does_not_equal_unbalanced_image_average(self):
        values=fixture();r=analyze(*values)
        self.assertEqual(len(r['by_attribute']),64)
        self.assertEqual(r['decision']['n_attributes'],16)
        self.assertEqual(r['decision']['contrasts']['all_attributes/correct_mean-shuffle_mean']['positive_attributes'],16)
        self.assertTrue(r['decision']['support_screen_passed'])

if __name__=='__main__':
    unittest.main(verbosity=2)
