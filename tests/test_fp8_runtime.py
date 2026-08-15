import unittest

from scripts.fp8_runtime import configure_triton_fp8_baseline


class Fp8Module:
    _deepgemm_disabled = False


class OtherModule:
    pass


class Model:
    def __init__(self, modules):
        self._modules = modules

    def modules(self):
        return iter(self._modules)


class ConfigureTritonFp8BaselineTests(unittest.TestCase):
    def test_disables_deepgemm_for_every_fp8_module(self) -> None:
        fp8_modules = [Fp8Module(), Fp8Module()]
        count = configure_triton_fp8_baseline(
            Model([OtherModule(), *fp8_modules]),
            module_types=(Fp8Module,),
        )

        self.assertEqual(count, 2)
        self.assertTrue(all(module._deepgemm_disabled for module in fp8_modules))

    def test_rejects_model_without_fp8_modules(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "contains no fine-grained FP8 modules"):
            configure_triton_fp8_baseline(
                Model([OtherModule()]),
                module_types=(Fp8Module,),
            )


if __name__ == "__main__":
    unittest.main()
