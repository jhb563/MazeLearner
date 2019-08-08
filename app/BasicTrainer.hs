module Main where

import BasicLearner
import WorldParser (loadWorldFromFile)

main :: IO ()
main = do
  world <- loadWorldFromFile "training_games/maze_grid_10_10_2_1.game"
  (winCount, finalWeights) <- playGameTraining world
  print winCount
  print finalWeights
