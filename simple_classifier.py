import numpy as np
from Dataset.datasets import boats_dataset

noise = np.zeros(128)
racer = np.zeros(128)
cam_sub = np.zeros(128)
sub = np.zeros(128)
tugboat = np.zeros(128)

stored_vals = {
            0: noise,
            1: racer,
            2: cam_sub,
            3: sub,
            4: tugboat,
            }

def classes(i):
    classes = {
            0: 'noise',
            1: 'racer',
            2: 'cam_sub',
            3: 'sub',
            4: 'tugboat'
            }
    return classes.get(i, "invalid boat")

def store_values(boat, row, row_matrix):
    if (classes(boat) == 'noise'):
        noise[row] += sum(sum(row_matrix))
    if (classes(boat) == 'racer'):
        racer[row] += sum(sum(row_matrix))
    if (classes(boat) == 'cam_sub'):
        cam_sub[row] += sum(sum(row_matrix))
    if (classes(boat) == 'sub'):
        sub[row] += sum(sum(row_matrix))
    if (classes(boat) == 'tugboat'):
        tugboat[row] += sum(sum(row_matrix))

def find_best_row(dataset):
    train_x, train_y, test_x, test_y = dataset
    num_noise = 0
    num_racer = 0
    num_sub_cam = 0
    num_sub = 0
    num_tugboat = 0
    i = 0
    for boat in train_y:
        count = 0
        if (boat == 0):
            num_noise += 1
        elif(boat == 1):
            num_racer += 1
        elif(boat == 2):
            num_sub_cam += 1
        elif(boat == 3):
            num_sub += 1
        elif(boat == 4):
            num_tugboat += 1
            
        for row in train_x[i]:
            store_values(boat, count, row)
            count+=1
        i += 1
    return num_noise, num_racer, num_sub_cam, num_sub, num_tugboat

def print_stored_vals():
    print(stored_vals)

def get_stored_vals_row(boat, row):
    return stored_vals[boat][row]

def get_max_val(boat):
    print(np.max(stored_vals[boat]))

num_noise, num_racer, num_sub_cam, num_sub, num_tugboat = find_best_row(boats_dataset())
get_stored_vals_row(0, 24)
get_max_val(1)
print(stored_vals[4])
print(np.mean(stored_vals[0]))

noise_mean = np.mean(stored_vals[0]) / num_noise
racer_mean = np.mean(stored_vals[1]) / num_racer
cam_sub_mean = np.mean(stored_vals[2]) / num_sub_cam
sub_mean = np.mean(stored_vals[3]) / num_sub
tugboat_mean = np.mean(stored_vals[4]) / num_tugboat

print("Noise avg: " + str(noise_mean) + " - " + str(num_noise) + " noise instances")
print("Racer avg: " + str(racer_mean) + " - " + str(num_racer) + " racer instances")
print("Sub_cam avg: " + str(cam_sub_mean) + " - " + str(num_sub_cam) + " cam_sub instances")
print("Sub avg: " + str(sub_mean) + " -rain " + str(num_sub) + " sub instances")
print("Tugboat avg: " + str(tugboat_mean) + " - " + str(num_tugboat) + " tugboat instances")

def predict_images(dataset):

    train_x, train_y, test_x, test_y = dataset
    i = 0
    correct = 0
    wrong = 0
    accuracy = 0.0
    noise_correct = 0
    racer_correct = 0
    sub_cam_correct = 0
    sub_correct = 0
    tugboat_correct = 0
    
    num_noise = 0
    num_racer = 0
    num_sub_cam = 0
    num_sub = 0
    num_tugboat = 0
    not_classified = 0
    
    # change train_y, and train_x to test_y, and test_x to make predictions on the test set.
    for boat in train_y:
        val = np.zeros(128)
        count = 0
        if (boat == 0):
            num_noise += 1
        elif(boat == 1):
            num_racer += 1
        elif(boat == 2):
            num_sub_cam += 1
        elif(boat == 3):
            num_sub += 1
        elif(boat == 4):
            num_tugboat += 1
        for row in train_x[i]:
            val[count] += sum(sum(row))
            count += 1
        avg_value = np.mean(val)
        
        if(avg_value < noise_mean + 12.843155):
            print("PREDICTED NOISE [" + str(avg_value) + "]- Actual: " + str(classes(boat)) + " " + str(noise_mean))
            if (boat == 0):
                correct += 1
                noise_correct += 1
            else:
                wrong += 1

        elif(avg_value > racer_mean - 10.606538318 and avg_value < racer_mean + 5.8859):
            print("PREDICTED RACER [" + str(avg_value) + "]- Actual: " + str(classes(boat)) + " " + str(racer_mean))
            if (boat == 1):
                correct += 1
                racer_correct += 1
            else:
                wrong += 1

        elif(avg_value > cam_sub_mean - 4.31 and avg_value < cam_sub_mean + 10.62):
            print("PREDICTED SUB_CAM [" + str(avg_value) + "]- Actual: " + str(classes(boat)) + " " + str(cam_sub_mean))
            if (boat == 2):
                correct += 1
                sub_cam_correct += 1
            else:
                wrong += 1

        elif(avg_value > sub_mean - 12.9 and avg_value < sub_mean + 4.32):
            print("PREDICTED SUB [" + str(avg_value) + "]- Actual: " + str(classes(boat)) + " " + str(sub_mean))
            if (boat == 3):
                correct += 1
                sub_correct += 1
            else:
                wrong += 1

        elif(avg_value > tugboat_mean - 5.9):
            print("PREDICTED TUGBOAT [" + str(avg_value) + "]- Actual: " + str(classes(boat)) + " " + str(tugboat_mean))
            if (boat == 4):
                correct += 1
                tugboat_correct += 1
            else:
                wrong += 1

        else:
            print("could not classify [" + str(avg_value) + "]")
            not_classified += 1
            wrong += 1
        i += 1
        
    accuracy = correct / (correct + wrong)
    print("\nNot classified: " + str(not_classified))
    print("Noise correctly classified: " + str(noise_correct) + " -> out of: " + str(num_noise))
    print("Racer correctly classified: " + str(racer_correct) + " -> out of: " + str(num_racer))
    print("Sub_cam correctly classified: " + str(sub_cam_correct) + " -> out of: " + str(num_sub_cam))
    print("Sub correctly classified: " + str(sub_correct) + " -> out of: " + str(num_sub))
    print("Tugboat correctly classified: " + str(tugboat_correct) + " -> out of: " + str(num_tugboat) + "\n")
    print("Correct predictions: " + str(correct))
    print("Wrong predictions: " + str(wrong))
    print("Accuracy: " + str(accuracy))
        
predict_images(boats_dataset())
