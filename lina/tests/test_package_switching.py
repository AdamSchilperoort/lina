import unittest


class TestPackageSwitching(unittest.TestCase):
    def test_lina_import_has_expected_surface(self):
        import lina

        for attr in (
            "utils",
            "props",
            "dm",
            "efc",
            "iefc",
            "aefc",
            "llowfsc",
            "set_backend",
            "get_backend",
        ):
            self.assertTrue(hasattr(lina, attr), f"lina missing attribute: {attr}")

    def test_lina_cpp_import_has_expected_surface(self):
        try:
            import lina_cpp
        except Exception as exc:
            raise unittest.SkipTest(f"lina_cpp unavailable: {exc}") from exc

        for attr in (
            "utils",
            "props",
            "dm",
            "efc",
            "iefc",
            "aefc",
            "llowfsc",
            "set_backend",
            "set_device",
            "get_device",
            "gpu_available",
        ):
            self.assertTrue(hasattr(lina_cpp, attr), f"lina_cpp missing attribute: {attr}")


if __name__ == "__main__":
    unittest.main()
