import matplotlib.pyplot as plt
from utility import get_SampleData, get_Model, get_Generator
from sklearn.model_selection import train_test_split

def main():

    EPOCHS = 5
    BATCH = 128
    ARCH = 2 # 1 for LeNet, 2 for NVIDIA
	
    #csv_file = '../simulator-self-driving-car/Data1/driving_log.csv'
    #csv_file = '../simulator-self-driving-car/Data2/driving_log.csv'
    #csv_file = '../simulator-self-driving-car/Data3_counterCLK/driving_log.csv'
    csv_file = './data/driving_log.csv'
	
    model = get_Model( ARCH )
    model.compile(loss='mse', optimizer='adam')
	
    # get data for training and validation
    train_samples, validation_samples = get_SampleData( csv_file )	
	
    train_generator = get_Generator(1, train_samples, BATCH) # 1 for training
    validation_generator = get_Generator(0, validation_samples, BATCH) # 0 for validation	
	
    history_object = model.fit_generator(train_generator, 
	                                     samples_per_epoch = (len(train_samples)//BATCH)*BATCH, 
                                         validation_data = validation_generator,
                                         nb_val_samples = len(validation_samples), 
                                         nb_epoch = EPOCHS)
                                         #verbose = 1)
	
    ### print the keys contained in the history object
    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()
	
    model.save('model.h5')	
	
if __name__ == '__main__':
    main()