import unittest
import s3_utils

class Test_S3_Utils(unittest.TestCase):

    def test_listing(self):
        files = list(s3_utils.list_files("s3://pillis-etl-data/AVIClips/"))
        self.assertGreater(len(files), 5)
