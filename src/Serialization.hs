module Serialization where

import qualified Data.Vector as V

import Player (WorldFeatures(..), produceWorldFeatures)
import Types

vectorizeWorld :: World -> V.Vector Float
vectorizeWorld w = V.fromList (fromIntegral <$>
  [ wfOnActiveEnemy features
  , wfShortestPathLength features
  , wfManhattanDistance features
  , wfEnemiesOnPath features
  , wfNearestEnemyDistance features
  , wfNumNearbyEnemies features
  , wfStunAvailable features
  , wfDrillsRemaining features
  ])
  where
    features = produceWorldFeatures w

moveFromOutput :: V.Vector Float -> PlayerMove
moveFromOutput vals = PlayerMove moveDirection useStun moveDirection
  where
    bestMoveIndex = V.maxIndex vals
    moveDirection = case bestMoveIndex `mod` 5 of
      0 -> DirectionUp
      1 -> DirectionRight
      2 -> DirectionDown
      3 -> DirectionLeft
      4 -> DirectionNone
    useStun = bestMoveIndex > 4
