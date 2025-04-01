# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 09:57:59 2025

@author: wsraymon
"""


class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class MaxIterSpotInitializationReachedError(Error):
    """Exception raised for spot initialization, while loop could not generate
    enough points to satisfy geometry constraints within 1 million tries.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message