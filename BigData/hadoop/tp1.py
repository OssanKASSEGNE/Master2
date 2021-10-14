import numpy
import utils.py
from "mixture.py" import Mixture
from pyspark.accumulators import AccumulatorParam

conf = SparkConf().setAppName(appName).setMaster(master)
sc = SparkContext(conf=conf)

hdf5file = sc.textFile("data_list.txt")


def load_cep(chaine):			

	chaine_tab = chaine.split(",")	#chaine(0] = filename ; chaine[1] = show
	
	return read_hdf5(hdf5file, show, dataset_list=("cep", "vad"))

def load_vad(chaine):			

	chaine_tab = chaine.split(",")	#chaine(0] = filename ; chaine[1] = show
	
	return read_hdf5(hdf5file, show, dataset_list=("cep", "vad"))


### Correction
