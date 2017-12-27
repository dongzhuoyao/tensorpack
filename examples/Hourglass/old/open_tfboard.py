#!/usr/bin/env mdl
# open the tensorboard
from config import config
import os, sys
import socket
import getpass
from setproctitle import setproctitle
import argparse

def print_warn(line):
    print('\033[93m '+ line+ ' \033[0m')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tb-log-dir', '-t', default=config.link_tensorboard_logger_log_dir)
    tb_dir = parser.parse_args().tb_log_dir
    port = config.tensorboard_port
    # log_dir = config.link_tensorboard_logger_log_dir
    if not os.path.isdir(tb_dir):
        print_warn('Please run the model first, and make sure config.tensorboard_enable=True!!')

    else:
        # tb_dir = config.link_tensorboard_logger_log_dir
        print('Tensorboard will monitor the data: ' + tb_dir)
        cmd_str = 'mdl python -m tensorflow.tensorboard --logdir ' + tb_dir
        host = socket.gethostname()
        user = getpass.getuser()

        ssh_tpl = 'ssh -L %d:' + '%s.%s.brc.sm.megvii-op.org' % (host, user) + ':%d' + ' -CAXY %s.%s.brc@brain.megvii-inc.com' % (host, user)
        setproctitle('skeleton')
        while True:
            tensorboard_cmd = cmd_str + ' --port ' + str(port)
            print_warn('Please use the following ssh_str to reconnect this vm, and use url "http://localhost:%d" to visit the tensorboard!' % port)
            print_warn(ssh_tpl % (port, port))
            ret = os.system(tensorboard_cmd)
            # if the system return, change the port
            # print('ret: ' + str(ret))
            if ret == 64768:
                print_warn('#' * 100)
                print_warn('Sorry, the port:%d has been occupied, please use the following ssh cmd to reconnect this vm!!!!')
                port += 1
            else:
                break
