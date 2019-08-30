{-# LANGUAGE DeriveGeneric #-}

module MoveRecordTypes where

import Data.Csv
import GHC.Generics

data MoveRecord114 = MoveRecord114
  { grid09 :: Float
  , grid19 :: Float
  , grid29 :: Float
  , grid39 :: Float
  , grid49 :: Float
  , grid59 :: Float
  , grid69 :: Float
  , grid79 :: Float
  , grid89 :: Float
  , grid99 :: Float
  , grid08 :: Float
  , grid18 :: Float
  , grid28 :: Float
  , grid38 :: Float
  , grid48 :: Float
  , grid58 :: Float
  , grid68 :: Float
  , grid78 :: Float
  , grid88 :: Float
  , grid98 :: Float
  , grid07 :: Float
  , grid17 :: Float
  , grid27 :: Float
  , grid37 :: Float
  , grid47 :: Float
  , grid57 :: Float
  , grid67 :: Float
  , grid77 :: Float
  , grid87 :: Float
  , grid97 :: Float
  , grid06 :: Float
  , grid16 :: Float
  , grid26 :: Float
  , grid36 :: Float
  , grid46 :: Float
  , grid56 :: Float
  , grid66 :: Float
  , grid76 :: Float
  , grid86 :: Float
  , grid96 :: Float
  , grid05 :: Float
  , grid15 :: Float
  , grid25 :: Float
  , grid35 :: Float
  , grid45 :: Float
  , grid55 :: Float
  , grid65 :: Float
  , grid75 :: Float
  , grid85 :: Float
  , grid95 :: Float
  , grid04 :: Float
  , grid14 :: Float
  , grid24 :: Float
  , grid34 :: Float
  , grid44 :: Float
  , grid54 :: Float
  , grid64 :: Float
  , grid74 :: Float
  , grid84 :: Float
  , grid94 :: Float
  , grid03 :: Float
  , grid13 :: Float
  , grid23 :: Float
  , grid33 :: Float
  , grid43 :: Float
  , grid53 :: Float
  , grid63 :: Float
  , grid73 :: Float
  , grid83 :: Float
  , grid93 :: Float
  , grid02 :: Float
  , grid12 :: Float
  , grid22 :: Float
  , grid32 :: Float
  , grid42 :: Float
  , grid52 :: Float
  , grid62 :: Float
  , grid72 :: Float
  , grid82 :: Float
  , grid92 :: Float
  , grid01 :: Float
  , grid11 :: Float
  , grid21 :: Float
  , grid31 :: Float
  , grid41 :: Float
  , grid51 :: Float
  , grid61 :: Float
  , grid71 :: Float
  , grid81 :: Float
  , grid91 :: Float
  , grid00 :: Float
  , grid10 :: Float
  , grid20 :: Float
  , grid30 :: Float
  , grid40 :: Float
  , grid50 :: Float
  , grid60 :: Float
  , grid70 :: Float
  , grid80 :: Float
  , grid90 :: Float
  , playerX :: Float
  , playerY :: Float
  , playerStun :: Float
  , playerDrills :: Float
  , e1X :: Float
  , e1Y :: Float
  , e1Stun :: Float
  , e2X :: Float
  , e2Y :: Float
  , e2Stun :: Float
  , d1X :: Float
  , d1Y :: Float
  , d2X :: Float
  , d2Y :: Float
  , resultingMove114 :: Float
  }
  deriving (Generic)

instance FromRecord MoveRecord114

data MoveRecord = MoveRecord
  { stillActiveEnemy :: Float
  , stillShortestPath :: Float
  , stillManhattanDistance :: Float
  , stillEnemiesOnPath :: Float
  , stillNearestEnemyDistance :: Float
  , stillNumNearbyEnemies :: Float
  , stillStunAvailable :: Float
  , stillDrillsRemaining :: Float
  , stillMoveEase :: Float
  , upActiveEnemy :: Float
  , upShortestPath :: Float
  , upManhattanDistance :: Float
  , upEnemiesOnPath :: Float
  , upNearestEnemyDistance :: Float
  , upNumNearbyEnemies :: Float
  , upStunAvailable :: Float
  , upDrillsRemaining :: Float
  , upMoveEase :: Float
  , rightActiveEnemy :: Float
  , rightShortestPath :: Float
  , rightManhattanDistance :: Float
  , rightEnemiesOnPath :: Float
  , rightNearestEnemyDistance :: Float
  , rightNumNearbyEnemies :: Float
  , rightStunAvailable :: Float
  , rightDrillsRemaining :: Float
  , rightMoveEase :: Float
  , downActiveEnemy :: Float
  , downShortestPath :: Float
  , downManhattanDistance :: Float
  , downEnemiesOnPath :: Float
  , downNearestEnemyDistance :: Float
  , downNumNearbyEnemies :: Float
  , downStunAvailable :: Float
  , downDrillsRemaining :: Float
  , downMoveEase :: Float
  , leftActiveEnemy :: Float
  , leftShortestPath :: Float
  , leftManhattanDistance :: Float
  , leftEnemiesOnPath :: Float
  , leftNearestEnemyDistance :: Float
  , leftNumNearbyEnemies :: Float
  , leftStunAvailable :: Float
  , leftDrillsRemaining :: Float
  , leftMoveEase :: Float
  , resultingMove :: Float
  }
  deriving (Generic)

instance FromRecord MoveRecord