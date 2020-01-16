import unittest
from environments import EnvTypes, EnvWrapperFactory


class TestEnvironmentCreation(unittest.TestCase):
    def test_CreateWindyGrid(self):
        env = EnvWrapperFactory(EnvTypes.WindyGridEnv)
        self.assertNotEqual(env, None)
        self.assertEqual(env.ActionSpaceN(), 4)
        self.assertEqual(env.ObservationSpaceN(), 70)

    def test_CreateTaxiGrid(self):
        env = EnvWrapperFactory(EnvTypes.TaxiGridEnv)
        self.assertNotEqual(env, None)
        self.assertEqual(env.ActionSpaceN(), 6)
        self.assertEqual(env.ObservationSpaceN(), 500)

    def test_CreateCartPole(self):
        env = EnvWrapperFactory(EnvTypes.CartPoleEnv)
        self.assertNotEqual(env, None)
        self.assertEqual(env.ActionSpaceN(), 2)
        self.assertEqual(env.ObservationSpaceN(), 4)

    def test_CreateAcroBot(self):
        env = EnvWrapperFactory(EnvTypes.AcroBotEnv)
        self.assertNotEqual(env, None)
        self.assertEqual(env.ActionSpaceN(), 3)
        self.assertEqual(env.ObservationSpaceN(), 6)

    def test_CreateMountainCar(self):
        env = EnvWrapperFactory(EnvTypes.MountainCarEnv)
        self.assertNotEqual(env, None)
        self.assertEqual(env.ActionSpaceN(), 3)
        self.assertEqual(env.ObservationSpaceN(), 2)

    def test_CreateContinuousMountainCar(self):
        env = EnvWrapperFactory(EnvTypes.ContinuousMountainCarEnv)
        self.assertNotEqual(env, None)
        self.assertEqual(env.ActionSpaceN(), 1)
        self.assertEqual(env.ObservationSpaceN(), 2)

    def test_CreateContinuousPendulum(self):
        env = EnvWrapperFactory(EnvTypes.ContinuousPendulumEnv)
        self.assertNotEqual(env, None)
        self.assertEqual(env.ActionSpaceN(), 1)
        self.assertEqual(env.ObservationSpaceN(), 3)


class TestEnvironmentWrapper(unittest.TestCase):
    # TODO: make these tests richer (i.e., more targeted checks)
    def CallEverythingInWrapper(self, envWrapper):
        self.assertIsNotNone(envWrapper.Reset())
        randomAction = envWrapper.env.action_space.sample()
        self.assertIsNotNone(envWrapper.Step(randomAction))
        envWrapper.Render()
        envWrapper.Close()

    def test_WindyGrid(self):
        self.CallEverythingInWrapper(EnvWrapperFactory(EnvTypes.WindyGridEnv))

    def test_TaxiGrid(self):
        self.CallEverythingInWrapper(EnvWrapperFactory(EnvTypes.TaxiGridEnv))

    def test_CartPole(self):
        self.CallEverythingInWrapper(EnvWrapperFactory(EnvTypes.CartPoleEnv))

    def test_AcroBot(self):
        self.CallEverythingInWrapper(EnvWrapperFactory(EnvTypes.AcroBotEnv))

    def test_MountainCar(self):
        self.CallEverythingInWrapper(EnvWrapperFactory(EnvTypes.MountainCarEnv))

    def test_ContinuousMountainCar(self):
        self.CallEverythingInWrapper(EnvWrapperFactory(EnvTypes.ContinuousMountainCarEnv))

    def test_ContinuousPendulum(self):
        self.CallEverythingInWrapper(EnvWrapperFactory(EnvTypes.ContinuousPendulumEnv))


if __name__ == "__main__":
    unittest.main()
