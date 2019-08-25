{-# LANGUAGE DeriveGeneric   #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE ScopedTypeVariables #-}

module SupervisedLearner where

import Control.Monad (when, forM_)
import Control.Monad.IO.Class (liftIO)
import Data.ByteString.Lazy.Char8 (pack)
import Data.Csv
import Data.Int (Int64)
import Data.Vector (Vector)
import qualified Data.Vector as V
import GHC.Generics
import System.Random.Shuffle
import TensorFlow.Core
import TensorFlow.Minimize
import TensorFlow.Ops hiding (initializedVariable, pack)
import TensorFlow.Variable

moveFeatures :: Int64
moveFeatures = 40

hiddenUnits :: Int64
hiddenUnits = 100

moveLabels :: Int64
moveLabels = 10

data MoveRecord = MoveRecord
  { upActiveEnemy :: Float
  , upShortestPath :: Float
  , upManhattanDistance :: Float
  , upEnemiesOnPath :: Float
  , upNearestEnemyDistance :: Float
  , upNumNearbyEnemies :: Float
  , upStunAvailable :: Float
  , upDrillsRemaining :: Float
  , rightActiveEnemy :: Float
  , rightShortestPath :: Float
  , rightManhattanDistance :: Float
  , rightEnemiesOnPath :: Float
  , rightNearestEnemyDistance :: Float
  , rightNumNearbyEnemies :: Float
  , rightStunAvailable :: Float
  , rightDrillsRemaining :: Float
  , downActiveEnemy :: Float
  , downShortestPath :: Float
  , downManhattanDistance :: Float
  , downEnemiesOnPath :: Float
  , downNearestEnemyDistance :: Float
  , downNumNearbyEnemies :: Float
  , downStunAvailable :: Float
  , downDrillsRemaining :: Float
  , leftActiveEnemy :: Float
  , leftShortestPath :: Float
  , leftManhattanDistance :: Float
  , leftEnemiesOnPath :: Float
  , leftNearestEnemyDistance :: Float
  , leftNumNearbyEnemies :: Float
  , leftStunAvailable :: Float
  , leftDrillsRemaining :: Float
  , stillActiveEnemy :: Float
  , stillShortestPath :: Float
  , stillManhattanDistance :: Float
  , stillEnemiesOnPath :: Float
  , stillNearestEnemyDistance :: Float
  , stillNumNearbyEnemies :: Float
  , stillStunAvailable :: Float
  , stillDrillsRemaining :: Float
  , resultingMove :: Float
  }
  deriving (Generic)

instance FromRecord MoveRecord

sampleSize :: Int
sampleSize = 100

readRecordFromFile :: FilePath -> IO (Vector MoveRecord)
readRecordFromFile fp = do
  contents <- pack <$> readFile fp
  let results = decode NoHeader contents :: Either String (Vector MoveRecord)
  case results of
    Left err -> error err
    Right records -> return records

chooseRandomRecords :: Vector MoveRecord -> IO (Vector MoveRecord)
chooseRandomRecords records = do
  let numRecords = V.length records
  chosenIndices <- take sampleSize <$> shuffleM [0..(numRecords - 1)]
  return $ V.fromList $ map (records V.!) chosenIndices

convertRecordsToTensorData :: Vector MoveRecord -> (TensorData Float, TensorData Int64)
convertRecordsToTensorData records = (input, output)
  where
    numRecords = V.length records
    input = encodeTensorData (Shape [fromIntegral numRecords, fromIntegral moveFeatures]) (V.fromList $ concatMap recordToInputs records)
    output = encodeTensorData (Shape [fromIntegral numRecords]) (round . resultingMove <$> records)
    recordToInputs :: MoveRecord -> [Float]
    recordToInputs rec =
      [ upActiveEnemy rec
      , upShortestPath rec
      , upManhattanDistance rec
      , upEnemiesOnPath rec
      , upNearestEnemyDistance rec
      , upNumNearbyEnemies rec
      , upStunAvailable rec
      , upDrillsRemaining rec
      , rightActiveEnemy rec
      , rightShortestPath rec
      , rightManhattanDistance rec
      , rightEnemiesOnPath rec
      , rightNearestEnemyDistance rec
      , rightNumNearbyEnemies rec
      , rightStunAvailable rec
      , rightDrillsRemaining rec
      , downActiveEnemy rec
      , downShortestPath rec
      , downManhattanDistance rec
      , downEnemiesOnPath rec
      , downNearestEnemyDistance rec
      , downNumNearbyEnemies rec
      , downStunAvailable rec
      , downDrillsRemaining rec
      , leftActiveEnemy rec
      , leftShortestPath rec
      , leftManhattanDistance rec
      , leftEnemiesOnPath rec
      , leftNearestEnemyDistance rec
      , leftNumNearbyEnemies rec
      , leftStunAvailable rec
      , leftDrillsRemaining rec
      , stillActiveEnemy rec
      , stillShortestPath rec
      , stillManhattanDistance rec
      , stillEnemiesOnPath rec
      , stillNearestEnemyDistance rec
      , stillNumNearbyEnemies rec
      , stillStunAvailable rec
      , stillDrillsRemaining rec
      ] 

buildNNLayer :: Int64 -> Int64 -> Tensor v Float -> Build (Variable Float, Variable Float, Tensor Build Float)
buildNNLayer inputSize outputSize input = do
  weights <- truncatedNormal (vector [inputSize, outputSize]) >>= initializedVariable
  bias <- truncatedNormal (vector [outputSize]) >>= initializedVariable
  let results = (input `matMul` readValue weights) `add` readValue bias
  return (weights, bias, results)

data Model = Model
  { train :: TensorData Float
          -> TensorData Int64
          -> Session ()
  , errorRate :: TensorData Float
              -> TensorData Int64
              -> Session (V.Vector Float)
  , w1 :: Variable Float
  , b1 :: Variable Float
  , w2 :: Variable Float
  , b2 :: Variable Float
  }

createModel :: Build Model
createModel = do
  let batchSize = -1
  (inputs :: Tensor Value Float) <- placeholder [batchSize, moveFeatures]
  (outputs :: Tensor Value Int64) <- placeholder [batchSize]
  (hiddenWeights, hiddenBiases, hiddenResults) <- buildNNLayer moveFeatures hiddenUnits inputs
  let rectifiedHiddenResults = relu hiddenResults
  (finalWeights, finalBiases, finalResults) <- buildNNLayer hiddenUnits moveLabels rectifiedHiddenResults
  (actualOutput :: Tensor Value Int64) <- render $ argMax finalResults (scalar (1 :: Int64))
  let (correctPredictions :: Tensor Build Float) = cast $ equal actualOutput outputs
  (errorRate_ :: Tensor Value Float) <- render $ 1 - (reduceMean correctPredictions)
  let outputVectors = oneHot outputs (fromIntegral moveLabels) 1 0
  let loss = reduceMean $ fst $ softmaxCrossEntropyWithLogits finalResults outputVectors
  let params = [hiddenWeights, hiddenBiases, finalWeights, finalBiases]
  train_ <- minimizeWith adam loss params
  return $ Model
    { train = \inputFeed outputFeed ->
        runWithFeeds
          [ feed inputs inputFeed
          , feed outputs outputFeed
          ]
          train_
    , errorRate = \inputFeed outputFeed ->
        runWithFeeds
          [ feed inputs inputFeed
          , feed outputs outputFeed
          ]
          errorRate_
    , w1 = hiddenWeights
    , b1 = hiddenBiases
    , w2 = finalWeights
    , b2 = finalBiases
    }

runTraining :: FilePath -> IO (Vector Float, Vector Float, Vector Float, Vector Float)
runTraining totalFile = runSession $ do
  initialRecords <- liftIO $ readRecordFromFile totalFile
  shuffledRecords <- liftIO $ shuffleM (V.toList initialRecords)
  let testRecords = V.fromList $ take 2000 shuffledRecords
  let trainingRecords = V.fromList $ drop 2000 shuffledRecords
  model <- build createModel

  forM_ ([0..50000] :: [Int]) $ \i -> do
    trainingSample <- liftIO $ chooseRandomRecords trainingRecords
    let (trainingInputs, trainingOutputs) = convertRecordsToTensorData trainingSample
    (train model) trainingInputs trainingOutputs
    when (i `mod` 100 == 0) $ do
      err <- (errorRate model) trainingInputs trainingOutputs
      liftIO $ putStrLn $ (show i) ++ " : current training error " ++ show ((err V.! 0) * 100)

  -- Testing
  let (testingInputs, testingOutputs) = convertRecordsToTensorData testRecords
  testingError <- (errorRate model) testingInputs testingOutputs
  liftIO $ putStrLn $ "test error " ++ show ((testingError V.! 0) * 100)

  w1' <- run (readValue $ w1 model)
  b1' <- run (readValue $ b1 model)
  w2' <- run (readValue $ w2 model)
  b2' <- run (readValue $ b2 model)
  return (w1', b1', w2', b2')
