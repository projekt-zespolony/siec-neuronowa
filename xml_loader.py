import xml.etree.ElementTree as et
import numpy as np
import statistics as stat


def normalize(data):
    mean = stat.mean(data)
    stdev = stat.stdev(data)

    return [((d - mean) / stdev) for d in data]

def acquire_normalized_data(xml_filename, partition_coeff):

    tree = et.parse(xml_filename)
    root = tree.getroot()

    timestamp = []
    temperature = []
    air_pressure = []
    humidity = []
    air_quality = []
    windows_opened = []
    people_in_the_room = []

    for c0 in root:
        for c1 in c0:
            if c1.tag == 'values':
                timestamp.append(c1.attrib.get('timestamp'))
                temperature.append(float(c1.attrib.get('temperature')))
                air_pressure.append(float(c1.attrib.get('airPressure')))
                air_quality.append(float(c1.attrib.get('humidity')))
                humidity.append(float(c1.attrib.get('airQuality')))
            else:
                windows_opened.append(float(c1.attrib.get('windowsOpened')))
                people_in_the_room.append(float(c1.attrib.get('peopleInTheRoom')))

    timestamp = [int(v[6:]) for v in timestamp]
    timestamp = normalize(timestamp)
    temperature = normalize(temperature)
    air_pressure = normalize(air_pressure)
    humidity = normalize(humidity)
    air_quality = normalize(air_quality)

    all_data = np.zeros((5, len(timestamp)))
    all_labels = np.zeros((2, len(timestamp)))

    for i in range(5):
        for j in range(len(timestamp)):
            if (i == 0):
                all_data[i][j] = timestamp[j]
                all_labels[i][j] = windows_opened[j]
            elif (i == 1):
                all_data[i][j] = temperature[j]
                all_labels[i][j] = people_in_the_room[j]
            elif (i == 2):
                all_data[i][j] = air_pressure[j]
            elif (i == 3):
                all_data[i][j] = humidity[j]
            else:
                all_data[i][j] = air_quality[j]
    
    all_data = np.transpose(all_data)
    all_labels = np.transpose(all_labels)

    border = int(len(all_data) * partition_coeff)

    train_data = all_data[:border]
    test_data = all_data[border:]

    train_labels = all_labels[:border]
    test_labels = all_labels[border:]

    return (train_data, train_labels), (test_data, test_labels)
