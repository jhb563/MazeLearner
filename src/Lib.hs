module Lib
    ( someFunc
    ) where

import TensorFlow.Core
import TensorFlow.Ops
import TensorFlow.GenOps.Core
import TensorFlow.Variable
import TensorFlow.Session
import TensorFlow.Minimize

import Player (evaluateWorld)

someFunc :: IO ()
someFunc = putStrLn "someFunc"
