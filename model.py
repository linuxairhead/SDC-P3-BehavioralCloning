

from utility import get_data, get_model

def main():

    csv_file = '../simulator-self-driving-car/Data2/driving_log.csv'
	#csv_file = '../simulator-self-driving-car/Data3_counterCLK/driving_log.csv'

    # get data for training and validation
    data = get_data( csv_file )

    architecture = 2 # 1 for LeNet, 2 for NVIDIA
	
    model = get_model( architecture )
    model.compile(loss='mse', optimizer='adam')
    model.fit(data[0], data[1], validation_split=0.2, shuffle=True, nb_epoch=6)
	
    model.save('model.h5')

if __name__ == '__main__':
    main()