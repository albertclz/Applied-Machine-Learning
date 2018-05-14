#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 00:28:47 2018

@author: albertzhang
"""
import sys

def test_length():
    if (sys.version_info>(3,0)):
        file = open('/Users/albertzhang/Desktop/18spring/Applied_ML/HW/HW_1/homework-1-albertzhangchi/task2/input.txt')
        text = file.readline().strip()
        assert len(text) == 6
    else:
        file = open('/Users/albertzhang/Desktop/18spring/Applied_ML/HW/HW_1/homework-1-albertzhangchi/task2/input.txt','rb')
        text = file.readline().strip().decode('utf-8')
        assert len(text) == 6
