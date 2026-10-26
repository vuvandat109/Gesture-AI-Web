import io
import unittest
import zipfile
import numpy as np
from sensor_model import read_motion, motion_features

class SensorInputTests(unittest.TestCase):
    def csv(self, label='up'):
        return ('acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,label,trial_id\n' + ''.join(f'{i},2,3,4,5,6,{label},one\n' for i in range(178))).encode()
    def test_labels_are_not_features(self):
        a,_=read_motion(self.csv('up'),'x.csv'); b,_=read_motion(self.csv('down'),'different.csv')
        np.testing.assert_array_equal(motion_features(a),motion_features(b))
    def test_missing_gyroscope_rejected(self):
        with self.assertRaises(ValueError): read_motion(b'ax,ay,az\n1,2,3\n','x.csv')
    def test_zip_aligns_axes_and_time(self):
        b=io.BytesIO()
        with zipfile.ZipFile(b,'w') as z:
            for name, offset in [('Accelerometer',0),('Gyroscope',10)]:
                z.writestr(name+'.csv','seconds_elapsed,z,y,x\n'+''.join(f'{i/100},3,2,{i+offset}\n' for i in range(178)))
        x,_=read_motion(b.getvalue(),'input.zip')
        np.testing.assert_allclose(x[:,0],np.arange(178)); np.testing.assert_allclose(x[:,3],np.arange(178)+10)
    def test_window_feature_shape(self):
        x,_=read_motion(self.csv(),'a.csv')
        self.assertEqual(motion_features(x).shape,(582,))

if __name__=='__main__': unittest.main()
