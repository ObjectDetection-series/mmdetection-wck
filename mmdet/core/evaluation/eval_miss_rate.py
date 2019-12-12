"""
Author: Yuan Yuan.
Date:2019/02/11.
Description: this file defines a function to start Matlab engine on background
             and perform miss rate evaluation.dbEval() is a srcipt for evaluating
             miss rate used in pedestrian detection.
             The related path should be set in Matlab path variable. (This word is important)
"""
import matlab.engine

def eval_caltech_mr():
    eng = matlab.engine.start_matlab()
    eng.dbEval(nargout=0)


def eval_kaist_mr():
    eng = matlab.engine.start_matlab()
    eng.kaist_eval(nargout=0)

def eval_cvc_mr():
    eng = matlab.engine.start_matlab()
    eng.cvc_eval(nargout=0)

