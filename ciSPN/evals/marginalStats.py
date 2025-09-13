import numpy as np


class MarginalStats:
    """
    Stats about regression performance
    """

    def __init__(self, num_vars):
        # num_vars is important for the average error, if that should describe the average error between a feature and
        # its prediction (see the ..._per_feature errors for information of the specific features)
        self.num_vars = num_vars

        self._error_l1 = 0
        self._error_l2 = 0
        self._correct = 0  # measured on whether 0 or 1 is more likely
        self._correct_elementwise = 0
        self._num_evals = 0

        self._error_l1_per_feature = None
        self._error_l2_per_feature = None

    def eval(self, expected, prediction):
        assert expected.shape == prediction.shape
        assert expected.shape[1] == self.num_vars
        # round to next class

        self._error_l1 += np.sum(np.abs(expected - prediction))
        self._error_l2 += np.sum(np.power(expected - prediction, 2))

        error_1_pf = np.sum(np.abs(expected - prediction), axis=0)
        if self._error_l1_per_feature is None:
            self._error_l1_per_feature = error_1_pf
        else:
            self._error_l1_per_feature += error_1_pf
        error_2_pf = np.sum(np.power(expected - prediction, 2), axis=0)
        if self._error_l2_per_feature is None:
            self._error_l2_per_feature = error_2_pf
        else:
            self._error_l2_per_feature += error_2_pf

        expected_0_1 = np.where(expected > 0.5, 1, 0)
        prediction_0_1 = np.where(prediction > 0.5, 1, 0)
        correct = np.all(expected_0_1 == prediction_0_1, axis=1)
        self._correct += np.sum(correct).item()
        correct_elementwise = np.sum(expected_0_1 == prediction_0_1)
        self._correct_elementwise += correct_elementwise.item()
        self._num_evals += len(expected)

        return correct

    def get_accuracy(self):
        if self._num_evals == 0:
            return -1
        return self._correct / self._num_evals

    def get_correct_elementwise(self):
        if self._num_evals == 0:
            return -1
        return self._correct_elementwise / (self._num_evals * self.num_vars)

    def get_error_l1(self):
        return self._error_l1
    
    def get_error_l2(self):
        return self._error_l2

    def get_error_l1_per_feature(self):
        return self._error_l1_per_feature

    def get_error_l2_per_feature(self):
        return self._error_l2_per_feature

    def get_eval_result_str(self):
        if self._num_evals == 0:
            return "No evaluations."
        return (
            f"Classified {self._num_evals} samples.\nCorrect: {self._correct}.\nAccuracy: {self.get_accuracy()}\n"
            f"Elementwise Accuracy: {self.get_correct_elementwise()}\n"
            f"Average error l1: {self.get_error_l1() / (self._num_evals * self.num_vars)}, "
            f"Average error l2: {self.get_error_l2() / (self._num_evals * self.num_vars)}"
        )

    def get_eval_result_str_per_feature(self):
        if self._num_evals == 0:
            return "No evaluations."
        str1 = " ".join(
            [f"{error/self._num_evals:.2f}" for error in self.get_error_l1_per_feature()]
        )
        str2 = " ".join(
            [f"{error/self._num_evals:.2f}" for error in self.get_error_l2_per_feature()]
        )
        return (
            f"Classified {self._num_evals} samples.\nAverage l1 error per feature:\n{str1}\n"
            f"Average l2 error per feature:\n{str2}"
        )
    
    def save_stats(self, path):
        np.savez(
            path,
            error_l1=self.get_error_l1(),
            error_l2=self.get_error_l2(),
            correct=self._correct,
            num_evals=self._num_evals,
            accuracy=self.get_accuracy(),
            correct_elementwise=self.get_correct_elementwise(),
            error_l1_per_feature=self.get_error_l1_per_feature(),
            error_l2_per_feature=self.get_error_l2_per_feature(),
        )
