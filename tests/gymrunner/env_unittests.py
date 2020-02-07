import unittest
from environments import *


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

    def test_CreateContinuousMountainCar(self):
        self.assertNotEqual(EnvWrapperFactory(EnvTypes.ContinuousMountainCarEnv), None)

    def test_CreateContinuousPendulum(self):
        self.assertNotEqual(EnvWrapperFactory(EnvTypes.ContinuousPendulumEnv), None)


class TestEnvironmentWrapper(unittest.TestCase):
    # TODO: make these tests richer (i.e., more targeted checks)
    def CallRoutineFunctions(self, envWrapper):
        self.assertIsNotNone(envWrapper.Reset())
        randomAction = envWrapper.env.action_space.sample()
        self.assertIsNotNone(envWrapper.Step(randomAction))
        envWrapper.Render()
        envWrapper.Close()

    def test_WindyGrid(self):
        ew = EnvWrapperFactory(EnvTypes.WindyGridEnv)
        self.CallRoutineFunctions(ew)
        self.assertEqual(ew.ActionSpaceN(), 4)
        self.assertEqual(ew.ObservationSpaceN(), 70)

    def test_TaxiGrid(self):
        ew = EnvWrapperFactory(EnvTypes.TaxiGridEnv)
        self.CallRoutineFunctions(ew)
        self.assertEqual(ew.ActionSpaceN(), 6)
        self.assertEqual(ew.ObservationSpaceN(), 500)
        ew.SetDiscretizationParams(np.array([2]), np.array([2, 2, 2, 2]))
        self.assertEqual(ew.DiscretizeAction(0), 0)
        self.assertEqual(ew.DiscretizeAction(1), 1)

    def test_CartPole(self):
        ew = EnvWrapperFactory(EnvTypes.CartPoleEnv)
        self.CallRoutineFunctions(ew)
        self.assertEqual(ew.ActionSpaceN(), 2)
        self.assertEqual(ew.ObservationSpaceN(), 4)
        ew.SetDiscretizationParams(np.array([2]), np.array([2, 2, 2, 2]))
        self.assertEqual(ew.DiscretizeAction(0), 0)
        self.assertEqual(ew.DiscretizeAction(1), 1)
        self.assertTrue(np.array_equal(ew.DiscretizeObservation([1, 1e-2, 1e-1, 1e-1]), [1, 1, 1, 1]))
        self.assertTrue(np.array_equal(ew.DiscretizeObservation([-1, -1e-2, -1e-1, -1e-1]), [0, 0, 0, 0]))
        # Note that this function should be doing a flooring operation for anything less than bin val
        self.assertTrue(np.array_equal(ew.DiscretizeObservation(ew.env.observation_space.high), [2, 2, 2, 2]))
        self.assertTrue(np.array_equal(ew.DiscretizeObservation(ew.env.observation_space.low), [0, 0, 0, 0]))


    def test_AcroBot(self):
        ew = EnvWrapperFactory(EnvTypes.AcroBotEnv)
        self.CallRoutineFunctions(ew)
        self.assertEqual(ew.ActionSpaceN(), 3)
        self.assertEqual(ew.ObservationSpaceN(), 6)

    def test_MountainCar(self):
        ew = EnvWrapperFactory(EnvTypes.MountainCarEnv)
        self.CallRoutineFunctions(ew)
        self.assertEqual(ew.ActionSpaceN(), 3)
        self.assertEqual(ew.ObservationSpaceN(), 2)

    def test_ContinuousMountainCar(self):
        ew = EnvWrapperFactory(EnvTypes.ContinuousMountainCarEnv)
        self.CallRoutineFunctions(ew)
        self.assertEqual(ew.ActionSpaceN(), 1)
        self.assertEqual(ew.ObservationSpaceN(), 2)

    def test_ContinuousPendulum(self):
        ew = EnvWrapperFactory(EnvTypes.ContinuousPendulumEnv)
        self.CallRoutineFunctions(ew)
        self.assertEqual(ew.ActionSpaceN(), 1)
        self.assertEqual(ew.ObservationSpaceN(), 3)


if __name__ == "__main__":
    unittest.main()
