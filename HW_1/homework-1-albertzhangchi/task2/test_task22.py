#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 23:47:25 2018

@author: albertzhang
"""
from __future__ import division
import numpy as np


# content of test_class.py
class TestClass(object):
    def test_one(self):
        assert 2/8 == 0.25

    def test_two(self):
        assert np.array([2])/np.array([8]) == 0.25