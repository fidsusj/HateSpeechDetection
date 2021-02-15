from unittest import TestCase

from path_helper import get_project_root


class Test(TestCase):
    def test_get_project_root(self):
        project_root_path = get_project_root()
        path_components = str(project_root_path).split("/")
        self.assertEqual("src", path_components[len(path_components) - 1])
        self.assertEqual(
            "HateSpeechDetection", path_components[len(path_components) - 2]
        )
