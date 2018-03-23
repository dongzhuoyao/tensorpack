# Author: Tao Hu <taohu620@gmail.com>
import numpy as np
import tensorflow as tf
from tensorpack.utils import logger
import six,copy
from tensorpack.tfutils.common import get_op_tensor_name
from tensorpack.tfutils.varmanip import (SessionUpdate, get_savename_from_varname,
                       is_training_name, get_checkpoint_path)

from tensorpack.tfutils.sessinit import DictRestore,SaverRestore, SessionInit, MismatchLogger

def my_get_model_loader(filename):
    """
    Get a corresponding model loader by looking at the file name.

    Returns:
        SessInit: either a :class:`DictRestore` (if name ends with 'npy/npz') or
        :class:`SaverRestore` (otherwise).
    """
    if filename.endswith('.npy'):
        assert tf.gfile.Exists(filename), filename
        return DictRestore(np.load(filename, encoding='latin1').item())
    elif filename.endswith('.npz'):
        assert tf.gfile.Exists(filename), filename
        obj = np.load(filename)
        return MyDictRestore(dict(obj))
    else:
        return SaverRestore(filename)

class MyDictRestore(SessionInit):
    """
    Restore variables from a dictionary.
    """

    def __init__(self, variable_dict):
        """
        Args:
            variable_dict (dict): a dict of {name: value}
        """
        assert isinstance(variable_dict, dict), type(variable_dict)
        # use varname (with :0) for consistency
        self._prms = {get_op_tensor_name(n)[1]: v for n, v in six.iteritems(variable_dict)}

    def _run_init(self, sess):
        total_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        for branch in ['support','query']:
            variables = [v for v in total_variables if branch in v.name]
            variable_names = set([k.name.replace("{}/".format(branch),"") for k in variables])
            param_names = set(six.iterkeys(self._prms))

            intersect = variable_names & param_names

            logger.info("Variables in branch-{} to restore from dict: {}".format(branch, ', '.join(map(str, intersect))))

            mismatch = MismatchLogger('graph', 'dict')
            for k in sorted(variable_names - param_names):
                if not is_training_name(k):
                    mismatch.add(k)
            mismatch.log()
            mismatch = MismatchLogger('dict', 'graph')
            for k in sorted(param_names - variable_names):
                mismatch.add(k)
            mismatch.log()

            upd = SessionUpdate(sess, [v for v in variables if v.name.replace("{}/".format(branch),"") in intersect])
            logger.info("Restoring branch-{}  from dict ...".format(branch))
            upd.update({"{}/{}".format(branch,name): value for name, value in six.iteritems(self._prms) if name.replace("{}/".format(branch),"") in intersect})
