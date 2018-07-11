package main

import (
	"boogie/learn"
	"io/ioutil"
	"fmt"
	"runtime"
	"time"
	"math/rand"
)

func getImages(path string) [][][]float64 {
	images := make([][][]float64, 10)
	for i := 0; i < 10; i++ {
		files, err := ioutil.ReadDir(fmt.Sprintf("%s/%d", path, i))
		if err != nil {
			fmt.Println(err)
			return nil
		}
		images[i] = make([][]float64, len(files))
		for j := 0; j < len(files); j++ {
			images[i][j] = learn.GetImageData(fmt.Sprintf("%s/%d/%s", path, i, files[j].Name()))
		}
	}
	return images
}

func main() {
	rand.Seed(time.Now().UnixNano())

	nn := learn.NewNeuralNet([]int{1024, 64, 10})
	nn.Load("nn.txt")

	trainingData := getImages("out/train")
	testingData := getImages("out/test")
	batchInputs := make([][]float64, 100)
	batchExpected := make([][]float64, 100)
	for batchNum := 0; true; batchNum++ {
		start := time.Now()
		for i := 0; i < 10; i++ {
			for j := 0; j < 10; j++ {
				batchInputs[10*i+j] = trainingData[i][rand.Intn(len(trainingData[i]))]
				batchExpected[10*i+j] = make([]float64, 10)
				batchExpected[10*i+j][i] = 1
			}
		}
		elapsed := time.Now().Sub(start)
		var m runtime.MemStats
		runtime.ReadMemStats(&m)
		start = time.Now()
		nn.Train(batchInputs, batchExpected)
		fmt.Printf("%d\t%s    \t%s\t%d\t%d\n", batchNum, elapsed, time.Now().Sub(start), m.Alloc>>14, m.NumGC)
		if batchNum%1000 == 0 {
			fmt.Println("---------------------------------------------")
			nn.Test(testingData)
			nn.Save("nn.txt")
			fmt.Println("---------------------------------------------")
		}
	}
}
