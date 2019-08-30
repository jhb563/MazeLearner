{-# LANGUAGE DeriveGeneric   #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE ScopedTypeVariables #-}

module SupervisedLearner where

import Control.Monad (when, forM_, forM)
import Control.Monad.State (StateT, evalStateT, get, put, lift)
import Control.Monad.IO.Class (liftIO)
import Data.ByteString.Lazy.Char8 (pack)
import Data.Csv
import Data.Int (Int64)
import Data.Vector (Vector)
import qualified Data.Vector as V
import System.IO
import System.Random (getStdGen)
import System.Random.Shuffle (shuffleM)
import TensorFlow.Core
import TensorFlow.Minimize
import TensorFlow.Ops hiding (initializedVariable, pack)
import TensorFlow.Variable

import Serialization (vectorizeWorld'')
import LearnerLib.Serialization (moveFromIndex, vectorizeWorld)
import MazeParser (generateRandomMaze)
import Runner (generateRandomWorld, stepWorld)
import Types
import MoveRecordTypes

moveFeatures :: Int64
moveFeatures = 114

hiddenUnits :: Int64
hiddenUnits = 100

moveLabels :: Int64
moveLabels = 10

sampleSize :: Int
sampleSize = 100

readRecordFromFile :: FilePath -> IO (Vector MoveRecord114)
readRecordFromFile fp = do
  contents <- pack <$> readFile fp
  let results = decode NoHeader contents :: Either String (Vector MoveRecord114)
  case results of
    Left err -> error err
    Right records -> return records

chooseRandomRecords :: Vector MoveRecord114 -> IO (Vector MoveRecord114)
chooseRandomRecords records = do
  let numRecords = V.length records
  chosenIndices <- take sampleSize <$> shuffleM [0..(numRecords - 1)]
  return $ V.fromList $ map (records V.!) chosenIndices

convertRecordsToTensorData :: Vector MoveRecord114 -> (TensorData Float, TensorData Int64)
convertRecordsToTensorData records = (input, output)
  where
    numRecords = V.length records
    input = encodeTensorData (Shape [fromIntegral numRecords, fromIntegral moveFeatures]) (V.fromList $ concatMap recordToInputs records)
    output = encodeTensorData (Shape [fromIntegral numRecords]) (round . resultingMove114 <$> records)
    recordToInputs :: MoveRecord114 -> [Float]
    recordToInputs rec =
      [ grid09 rec
      , grid19 rec
      , grid29 rec
      , grid39 rec
      , grid49 rec
      , grid59 rec
      , grid69 rec
      , grid79 rec
      , grid89 rec
      , grid99 rec
      , grid08 rec
      , grid18 rec
      , grid28 rec
      , grid38 rec
      , grid48 rec
      , grid58 rec
      , grid68 rec
      , grid78 rec
      , grid88 rec
      , grid98 rec
      , grid07 rec
      , grid17 rec
      , grid27 rec
      , grid37 rec
      , grid47 rec
      , grid57 rec
      , grid67 rec
      , grid77 rec
      , grid87 rec
      , grid97 rec
      , grid06 rec
      , grid16 rec
      , grid26 rec
      , grid36 rec
      , grid46 rec
      , grid56 rec
      , grid66 rec
      , grid76 rec
      , grid86 rec
      , grid96 rec
      , grid05 rec
      , grid15 rec
      , grid25 rec
      , grid35 rec
      , grid45 rec
      , grid55 rec
      , grid65 rec
      , grid75 rec
      , grid85 rec
      , grid95 rec
      , grid04 rec
      , grid14 rec
      , grid24 rec
      , grid34 rec
      , grid44 rec
      , grid54 rec
      , grid64 rec
      , grid74 rec
      , grid84 rec
      , grid94 rec
      , grid03 rec
      , grid13 rec
      , grid23 rec
      , grid33 rec
      , grid43 rec
      , grid53 rec
      , grid63 rec
      , grid73 rec
      , grid83 rec
      , grid93 rec
      , grid02 rec
      , grid12 rec
      , grid22 rec
      , grid32 rec
      , grid42 rec
      , grid52 rec
      , grid62 rec
      , grid72 rec
      , grid82 rec
      , grid92 rec
      , grid01 rec
      , grid11 rec
      , grid21 rec
      , grid31 rec
      , grid41 rec
      , grid51 rec
      , grid61 rec
      , grid71 rec
      , grid81 rec
      , grid91 rec
      , grid00 rec
      , grid10 rec
      , grid20 rec
      , grid30 rec
      , grid40 rec
      , grid50 rec
      , grid60 rec
      , grid70 rec
      , grid80 rec
      , grid90 rec
      , playerX rec
      , playerY rec
      , playerStun rec
      , playerDrills rec
      , e1X rec
      , e1Y rec
      , e1Stun rec
      , e2X rec
      , e2Y rec
      , e2Stun rec
      , d1X rec
      , d1Y rec
      , d2X rec
      , d2Y rec
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

data PlayModel = PlayModel
  { getMove :: TensorData Float -> Session (V.Vector Int64) }

createPlayModel :: (V.Vector Float, V.Vector Float, V.Vector Float, V.Vector Float) -> Build PlayModel
createPlayModel (w1, b1, w2, b2) = do
  inputFeatures <- placeholder [moveFeatures]
  let w1' = constant (Shape [moveFeatures, hiddenUnits]) (V.toList w1)
  let b1' = constant (Shape [hiddenUnits]) (V.toList b1)
  let w2' = constant (Shape [hiddenUnits, moveLabels]) (V.toList w2)
  let b2' = constant (Shape [moveLabels]) (V.toList b2)
  let hiddenResults = relu $ (inputFeatures `matMul` w1') `add` b1'
  let finalResults = (hiddenResults `matMul` w2') `add` b2'
  let outputMove = argMax finalResults (scalar (1 :: Int64))
  return $ PlayModel
    { getMove = \inputFeed -> runWithFeeds [ feed inputFeatures inputFeed ] outputMove }

readWeightsAndBiases :: FilePath -> IO (V.Vector Float, V.Vector Float, V.Vector Float, V.Vector Float)
readWeightsAndBiases fp = do
  handle <- openFile fp ReadMode
  w1 <- read <$> hGetLine handle
  b1 <- read <$> hGetLine handle
  w2 <- read <$> hGetLine handle
  b2 <- read <$> hGetLine handle
  hClose handle
  return (w1, b1, w2, b2)

runWorldIteration :: PlayModel -> StateT World Session Bool
runWorldIteration model = do
  w <- get
  let worldData = encodeTensorData (Shape [1, 114]) (vectorizeWorld'' w)
  moveIndex <- lift $ (getMove model) worldData
  let nextMove = moveFromIndex (fromIntegral $ V.head moveIndex)
  let (nextWorld, _) = stepWorld nextMove w
  put nextWorld
  case (worldResult nextWorld) of
    GameInProgress -> runWorldIteration model
    GameWon -> return True
    GameLost -> return False

playGameIterations :: FilePath -> Session Int
playGameIterations fp = do
  (w1, b1, w2, b2) <- liftIO $ readWeightsAndBiases fp
  model <- build $ createPlayModel (w1, b1, w2, b2)
  let gameParams = defaultGameParameters { numEnemies = 2, numDrillPowerups = 2 }
  results <- forM [1..100] $ \i -> do
    liftIO $ print i
    gen <- liftIO getStdGen
    let (randomMaze, gen') = generateRandomMaze gen (10, 10)
    let w = generateRandomWorld gameParams gen'
    let w' = w { worldBoundaries = randomMaze }
    evalStateT (runWorldIteration model) w'
  return $ length (filter id results)

playGameTraining :: FilePath -> IO Int
playGameTraining fp = runSession (playGameIterations fp)
