module Main where

import BasicLearner
import WorldParser (loadWorldFromFile)

main :: IO ()
main = do
  world <- loadWorldFromFile "training_games/empty_grid_5_5_1_0.game"
  (winCount, finalWeights) <- playGameTraining world
  print winCount
  print finalWeights
