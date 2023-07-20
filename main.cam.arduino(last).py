import cv2
import time
# 웹캠 캡처 객체 생성
capture = cv2.VideoCapture(0)  # 웹캠 인덱스: 0 (기본 웹캠)

# 비디오 코덱 설정
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# 저장할 비디오 파일명
output_filename = './test/output_video.avi'

# 비디오 저장을 위한 VideoWriter 객체 생성
output = cv2.VideoWriter(output_filename, fourcc, 20.0, (640, 480))  # 적절한 해상도와 FPS 설정
while True:
    
    #start_time = time.time()  # 시작 시간 기록
    
    # 비디오 프레임 읽기
    ret, frame = capture.read()

    # 캡처가 정상적으로 되었는지 확인
    if not ret:
        break

    # 프레임을 출력 비디오에 기록
    output.write(frame)
    
    
    
    # 화면에 프레임 출력
    cv2.imshow('Video', frame)
    
        

    
    #last_time=time.time()
    # 현재 시간과 시작 시간의 차이를 계산하여 3초가 경과하면 종료
    #if last_time - start_time >= 3:
    #    break
    # 'q' 키를 누르면 루프 종료
    if cv2.waitKey(4) & 0xFF == ord('q'):
        break
 
# 사용한 객체들 해제
capture.release()
output.release()
cv2.destroyAllWindows()












# !pip install -q git+https://github.com/tensorflow/docs
import serial
arduino = serial.Serial('COM4', 9600)  # Replace 'COM3' with the appropriate port and baud rate

# Setup
from tensorflow_docs.vis import embed
from tensorflow import keras
from imutils import paths

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import imageio
#import cv2
import os

# Define hyperparameters
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 1

# MAX_SEQ_LENGTH = 20
MAX_SEQ_LENGTH = 1280
NUM_FEATURES = 2048

# Data preparation
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print(f"Total videos for training: {len(train_df)}")
print(f"Total videos for testing: {len(test_df)}")

# ValueError: Cannot take a larger sample than population when 'replace=False'
#train_df.sample(10)


# The following two methods are taken from this tutorial:
# https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub


def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]


def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)
 
            #if len(frames) == max_frames:
             #   break

    # csh 5000밀리세컨드=5초 초과하면 브레이크, 영상 0~5초까지만
            if cap.get(cv2.CAP_PROP_POS_MSEC) > 5000:
                #print('5초 초과')
                break

    finally:
        cap.release()
    return np.array(frames)



# Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/inception_v3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5
def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")


feature_extractor = build_feature_extractor()


#
label_processor = keras.layers.StringLookup(
    num_oov_indices=0, vocabulary=np.unique(train_df["tag"])
)
print(label_processor.get_vocabulary())







#
def prepare_all_videos(df, root_dir):
    num_samples = len(df)
    video_paths = df["video_name"].values.tolist()
    labels = df["tag"].values
    labels = label_processor(labels[..., None]).numpy()

    # `frame_masks` and `frame_features` are what we will feed to our sequence model.
    # `frame_masks` will contain a bunch of booleans denoting if a timestep is
    # masked with padding or not.
    frame_masks = np.zeros(shape=(num_samples, MAX_SEQ_LENGTH), dtype="bool")
    print('frame_mask : ',frame_masks)
    frame_features = np.zeros(
        shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
    )
    print('frame_features: ',frame_features)
    # For each video.
    for idx, path in enumerate(video_paths):
        # Gather all its frames and add a batch dimension.
        frames = load_video(os.path.join(root_dir, path))
        print('frames : ',frames)
        frames = frames[None, ...]

        # Initialize placeholders to store the masks and features of the current video.
        temp_frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
        temp_frame_features = np.zeros(
            shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
        )

        # Extract features from the frames of the current video.
        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(MAX_SEQ_LENGTH, video_length)

            ###
            for j in range(length):
                # 1/1 ==============================
                temp_frame_features[i, j, :] = feature_extractor.predict(
                    batch[None, j, :]
                )
            ###

            temp_frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

        frame_features[idx,] = temp_frame_features.squeeze()
        frame_masks[idx,] = temp_frame_mask.squeeze()

    return (frame_features, frame_masks), labels

# 1/1 ================================== prepare_all_videos()
# 
train_data, train_labels = prepare_all_videos(train_df, "train")
test_data, test_labels = prepare_all_videos(test_df, "test")

print(f"Frame features in train set: {train_data[0].shape}")
print(f"Frame masks in train set: {train_data[1].shape}")


##################################################
##################################################
##################################################
##################################################
##################################################
##################################################







# Model
# Utility for our sequence model.
def get_sequence_model():
    
    #라벨을 가져옴 class_vocab
    class_vocab = label_processor.get_vocabulary()

    frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")

    # Refer to the following tutorial to understand the significance of using `mask`:
    # https://keras.io/api/layers/recurrent_layers/gru/
    x = keras.layers.GRU(16, return_sequences=True)(
        frame_features_input, mask=mask_input
    )
    x = keras.layers.GRU(8)(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(8, activation="relu")(x)
    output = keras.layers.Dense(len(class_vocab), activation="softmax")(x)

    rnn_model = keras.Model([frame_features_input, mask_input], output)

    rnn_model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return rnn_model

# Accuracy
# Utility for running experiments.
def run_experiment():
    # tmp폴더에 video_classifier (이건 폴더path 아님) / video_classifier는 파일이름 . 앞에 들어가는
    # 코랩에선 "/content"로 시작 O 
    # vscode에선 "./tmp"로 시작 O    "/tmp"로 시작 X
    # tmp 폴더 안만들어 놓고 돌려도 알아서 생성함
    filepath = "./tmp/video_classifier"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=False, save_best_only=True, verbose=1
    )

    seq_model = get_sequence_model()
    history = seq_model.fit(
        [train_data[0], train_data[1]],
        train_labels,
        
        validation_split=0.3,
        epochs=EPOCHS,
        callbacks=[checkpoint],
    )

    seq_model.load_weights(filepath)
    _, accuracy = seq_model.evaluate([test_data[0], test_data[1]], test_labels)

# csh 테스트아큐러시 파일로 저장
    #with open('testAccuracy.txt', 'a') as f:
        #print(f"Test accuracy: {round(accuracy * 100, 2)}%", file=f)
    

    return history, seq_model

_, sequence_model = run_experiment()








# inference
def prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH, ), dtype="bool")
    frame_featutes = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES),
                            dtype="float32")
    
    
    for i, batch in enumerate(frames):  
        video_length = batch.shape[0]
        #video_length = batch.shape[1]
        length = min(MAX_SEQ_LENGTH, video_length)  
        for j in range(length):
            # 1/1 ==================== feature_extractor.predict()
            frame_featutes[i, j, :] = feature_extractor.predict(batch[None, j, :])  
        frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

    return frame_featutes, frame_mask

def sequence_prediction(path):
    class_vocab = label_processor.get_vocabulary()

    frames = load_video(os.path.join("test", path))
    # 1/1 ==========prepare_single_video() > feature_extractor.predict()
    frame_features, frame_mask = prepare_single_video(frames)
    probabilities = sequence_model.predict([frame_features, frame_mask])[0]

    for i in np.argsort(probabilities)[::-1]:
        # csh w쓰기모드 a추가모드, 단일 개체 프레딕트 출력 
        with open('predict.txt', 'a') as f:
            
            # arduino
            prd = f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%"
            if (("rain" or "snow") in prd ) and  (float(prd[-6:-1]) >= 5) :
                arduino.write(b'pistol detected\n')                
            
            print(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%", file=f)
    return frames









# This utility is for visualization. 
# Referenced from:
# https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub
def to_gif(images):
    converted_images = images.astype(np.uint8)
    
    imageio.mimsave("animation.gif", converted_images, fps=10)
    return embed.embed_file("animation.gif")
# csh gif고정이름에서 변수로
#에러
    #imageio.mimsave(f"{converted_images[0:5]}.gif", converted_images, fps=10)
    #return embed.embed_file(f"{converted_images[0:5]}.gif")
#
# test_video = np.random.choice(test_df["video_name"].values.tolist())
# print(f"Test video path: {test_video}")

test_video0 = test_df["video_name"].values.tolist()[0] #normal
##test_video1 = test_df["video_name"].values.tolist()[1] #rain
#test_video2 = test_df["video_name"].values.tolist()[2] #snow
print(f"Test video path: {test_video0}")
#print(f"Test video path: {test_video1}")
#print(f"Test video path: {test_video2}")
test_frames0=sequence_prediction(test_video0)
#test_frames1=sequence_prediction(test_video1)
#test_frames2=sequence_prediction(test_video2)

# predict
# test_frames = sequence_prediction(test_video)

# convert to gif
#to_gif(test_frames0[:MAX_SEQ_LENGTH])
#to_gif(test_frames1[:MAX_SEQ_LENGTH])
#to_gif(test_frames2[:MAX_SEQ_LENGTH])
