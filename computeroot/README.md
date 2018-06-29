# computeroot

## Deployment

  - https://github.com/llSourcell/how_to_deploy_a_keras_model_to_production/blob/master/app.py
  - https://medium.com/@burgalon/deploying-your-keras-model-35648f9dc5fb
  - https://gitlab.com/fast-science/background-removal-server
  - https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html
  - https://github.com/mtobeiyf/keras-flask-deploy-webapp


- https://en.wikipedia.org/wiki/Universal_Chess_Interface
- https://github.com/zhelyabuzhsky/stockfish
- https://github.com/Dani4kor/stockfishpy


https://docs.python.org/2/library/subprocess.html

Square classifier:
Epoch 63/100
156/156 [==============================] - 15s 94ms/step - loss: 0.0928 - acc: 0.9700 - val_loss: 0.0598 - val_acc: 0.9884

Deeper Square classifier
Epoch 28/100
156/156 [==============================] - 9s 60ms/step - loss: 0.0578 - acc: 0.9839 - val_loss: 0.0584 - val_acc: 0.9868

Board Extractor:

Epoch 20/100
13/13 [==============================] - 175s 13s/step - loss: 0.0739 - dice_coeff: 0.9705 - val_loss: 0.0764 - val_dice_coeff: 0.9703

Board extraction:
  CPU: 593s 49s/step
  GPU: 175s 13s/step

Square classifier: 
  CPU: 75s 480ms/step
  GPU: 15s 94ms/step

~4x faster training!

FEN: 6 fields:
- piece placement (from white's perspective)
- Active color (w/b)
- Castling - KQkq or -
- En passent target square or -
- halfmove clock (start at 0)
- fullmove clock (start at 1)


### Architecture 

2 nodes: 
  - web node (serves static content)
  - compute node (handles POST requests to classify boards)

As more user data is uploaded to the compute node, the data is stored to further improve the models
The compute node follows roughly the following flow on user upload and user feedback

On new image: POST {file: "..."} to /cv_algo (checked on frontend)
  - give image unique_id
  - extract board
  if success:
    - save raw in ./user_uploads/raw_success/
  else:
    - save raw in ./user_uploads/raw_fail/
    - return {error: "msg"} to user
  - classify pieces (+ logic)
  - return {result: "fen", id: "..."} to user (front end produces feedback button)
  - save board in ./user_uploads/unlabelled/boards/
  - save predictions in ./user_uploads/unlabelled/predictions/ (same id + .json)

On new feedback event: POST to /feedback {id: "...", correct: true}
  if correct:
    - save all squares in board from ./user_uploads/unlabelled/boards/id to ./user_uploads/squares/<b, n, ...>/
  else: 
    - copy board from ./user_uploads/unlabelled/boards/id ./user_uploads/fail/boards/id
    - copy prediction from ./user_uploads/unlabelled/predictions/id ./user_uploads/fail/predictions/id
    - delete board from ./user_uploads/unlabelled/boards/id


    https://github.com/zhelyabuzhsky/stockfish