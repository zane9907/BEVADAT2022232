
import numpy as np


# Írj egy olyan fügvényt, ami megfordítja egy 2d array oszlopait
# Be: [[1,2],[3,4]]
# Ki: [[2,1],[4,3]]
# column_swap()


def column_swap(array:np.array) -> np.array:
    return np.fliplr(array)


# Készíts egy olyan függvényt ami összehasonlít két array-t és adjon vissza egy array-ben, hogy hol egyenlőek 
# Pl Be: [7,8,9], [9,8,7] 
# Ki: [1]
# compare_two_array()
# egyenlő elemszámúakra kell csak hogy működjön


def compare_two_array(arr1:np.array, arr2:np.array) -> np.array:
    equal_indices = np.where(arr1 == arr2)[0]
    return equal_indices


#Készíts egy olyan függvényt, ami vissza adja a megadott array dimenzióit:
# Be: [[1,2,3], [4,5,6]]
# Ki: "sor: 2, oszlop: 3, melyseg: 1"
# get_array_shape()
# 3D-vel még műküdnie kell!


def get_array_shape(arr:np.array):
    shape = np.shape(arr)
    if len(shape) == 1:
        return "sor: {}, oszlop: {}, melyseg: 1".format(shape[0], 1, 1)
    elif len(shape) == 2:
        return "sor: {}, oszlop: {}, melyseg: 1".format(shape[0], shape[1], 1)
    elif len(shape) == 3:
        return "sor: {}, oszlop: {}, melyseg: {}".format(shape[0], shape[1], shape[2])
    else:
        return "A tömb dimenziója nem támogatott"



# Készíts egy olyan függvényt, aminek segítségével elő tudod állítani egy neurális hálózat tanításához szükséges Y-okat egy numpy array-ből. 
#Bementként add meg az array-t, illetve hogy mennyi class-od van. Kimenetként pedig adjon vissza egy 2d array-t, ahol a sorok az egyes elemek. Minden nullákkal teli legyen és csak ott álljon egyes, ahol a bementi tömb megjelöli
# Be: [1, 2, 0, 3], 4
# Ki: [[0,1,0,0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
# encode_Y()


def encode_Y(arr:np.array, num_classes) ->np.array:
    encoded_arr = np.zeros((len(arr), num_classes))
    for i in range(len(arr)):
        encoded_arr[i][arr[i]] = 1
    return encoded_arr


# A fenti feladatnak valósítsd meg a kiértékelését. Adj meg a 2d array-t és adj vissza a decodolt változatát
# Be:  [[0,1,0,0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
# Ki:  [1, 2, 0, 3]
# decode_Y()


def decode_Y(y:np.array) ->np.array:
    decoded_y = np.argmax(y, axis=1)
    return decoded_y


# Készíts egy olyan függvényt, ami képes kiértékelni egy neurális háló eredményét! Bemenetként egy listát és egy array-t és adja vissza a legvalószínübb element a listából.
# Be: ['alma', 'körte', 'szilva'], [0.2, 0.2, 0.6]. 
# Ki: 'szilva'
# eval_classification()


def eval_classification(labels, predictions:np.array):
    max_index = np.argmax(predictions)
    return labels[max_index]


# Készíts egy olyan függvényt, ahol az 1D array-ben a páratlan számokat -1-re cseréli
# Be: [1,2,3,4,5,6]
# Ki: [-1,2,-1,4,-1,6]
# repalce_odd_numbers()


def replace_odd_numbers(arr:np.array):
    new_arr = np.copy(arr)
    new_arr[arr % 2 == 1] = -1
    
    return new_arr


# Készíts egy olyan függvényt, ami egy array értékeit -1 és 1-re változtatja, attól függően, hogy az adott elem nagyobb vagy kisebb a paraméterként megadott számnál.
# Ha a szám kisebb mint a megadott érték, akkor -1, ha nagyobb vagy egyenlő, akkor pedig 1.
# Be: [1, 2, 5, 0], 2
# Ki: [-1, 1, 1, -1]
# replace_by_value()


def replace_by_value(arr:np.array, value):
    arr = np.where(arr < value, -1, 1)
    return arr


# Készítsd egy olyan függvényt, ami az array értékeit összeszorozza és az eredmény visszaadja
# Be: [1,2,3,4]
# Ki: 24
# array_multi()
# Ha több dimenziós a tömb, akkor az egész tömb elemeinek szorzatával térjen vissza


def array_multi(arr:np.array):
    return np.prod(arr)


# Készítsd egy olyan függvényt, ami a 2D array értékeit összeszorozza és egy olyan array-el tér vissza, aminek az elemei a soroknak a szorzata
# Be: [[1, 2], [3, 4]]
# Ki: [2, 12]
# array_multi_2d()


def array_multi_2d(arr:np.array):
    return np.product(arr, axis=1)


# Készíts egy olyan függvényt, amit egy meglévő numpy array-hez készít egy bordert nullásokkal. Bementként egy array-t várjon és kimenetként egy array jelenjen meg aminek van border-je
# Be: [[1,2],[3,4]]
# Ki: [[0,0,0,0],[0,1,2,0],[0,3,4,0],[0,0,0,0]]
# add_border()



def add_border(arr):
    rows, cols = arr.shape
    new_arr = np.zeros((rows + 2, cols + 2))
    new_arr[1:-1, 1:-1] = arr
    
    return new_arr


# Készíts egy olyan függvényt ami két dátum között felsorolja az összes napot és ezt adja vissza egy numpy array-ben. A fgv ként str vár paraméterként 'YYYY-MM' formában.
# Be: '2023-03', '2023-04'  # mind a kettő paraméter str.
# Ki: ['2023-03-01', '2023-03-02', .. , '2023-03-31',]
# list_days()


def list_days(start_date, end_date):
    start = np.datetime64(start_date)
    end = np.datetime64(end_date)
    days = np.arange(start, end + np.timedelta64(1, 'D'), np.timedelta64(1, 'D'))
    return np.array([str(day) for day in days])


# Írj egy fügvényt ami vissza adja az aktuális dátumot az alábbi formában: YYYY-MM-DD. Térjen vissza egy 'numpy.datetime64' típussal.
# Be:
# Ki: 2017-03-24
# get_act_date()


def get_act_date():
    return np.datetime64('today')


# Írj egy olyan függvényt ami visszadja, hogy mennyi másodperc telt el 1970 január 01. 00:02:00 óta. Int-el térjen vissza
# Be: 
# Ki: másodpercben az idó, int-é kasztolva
# sec_from_1970()


def sec_from_1970():
    start_time = np.datetime64('1970-01-01T00:02:00')
    current_time = np.datetime64('now')
    elapsed_time = (current_time - start_time) / np.timedelta64(1, 's')
    return int(elapsed_time)


