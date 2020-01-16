import unittest
from environments import EnvTypes, EnvWrapperFactory


class TestEnvironmentCreation(unittest.TestCase):
    def test_CreateWindyGrid(self):
        self.assertNotEqual(EnvWrapperFactory(EnvTypes.WindyGridEnv), None)

    def test_CreateTaxiGrid(self):
        self.assertNotEqual(EnvWrapperFactory(EnvTypes.TaxiGridEnv), None)

    def test_CreateCartPole(self):
        self.assertNotEqual(EnvWrapperFactory(EnvTypes.CartPoleEnv), None)

    def test_CreateAcroBot(self):
        self.assertNotEqual(EnvWrapperFactory(EnvTypes.AcroBotEnv), None)

    def test_CreateMountainCar(self):
        self.assertNotEqual(EnvWrapperFactory(EnvTypes.MountainCarEnv), None)

# TODO: test other api (get size of action/observation space)


if __name__ == "__main__":
    unittest.main()
