package main

// TODO numpy would make the operations on multidimensial slices less painful
import (
	"encoding/gob"
	"fmt"
	"image/png"
	"io/ioutil"
	"math"
	"math/rand"
	"os"
	"time"
)

type neuralNet struct {
	Weights   [][][]float64
	Biases    [][]float64
	nodes     [][]float64 // the rest are only used during processing, but are part of struct for convenience
	activ     [][]float64
	wPartials [][][]float64
	bPartials [][]float64
	cPartials [][]float64
}

func newNeuralNet(layers []int) neuralNet {
	nodes := make([][]float64, len(layers))
	activ := make([][]float64, len(layers)-1)
	biases := make([][]float64, len(layers)-1)
	weights := make([][][]float64, len(layers)-1)
	wPartials := make([][][]float64, len(layers)-1)
	bPartials := make([][]float64, len(layers)-1)
	cPartials := make([][]float64, len(layers)-1)

	nodes[0] = make([]float64, layers[0])
	for i := 0; i < len(nodes)-1; i++ {
		nodes[i+1] = make([]float64, layers[i+1])
		activ[i] = make([]float64, layers[i])
		biases[i] = make([]float64, layers[i])
		weights[i] = make([][]float64, layers[i+1])
		wPartials[i] = make([][]float64, layers[i+1])
		bPartials[i] = make([]float64, layers[i+1])
		cPartials[i] = make([]float64, layers[i+1])
		for j := 0; j < layers[i+1]; j++ {
			weights[i][j] = make([]float64, layers[i])
			wPartials[i][j] = make([]float64, layers[i])
			for k := 0; k < layers[i]; k++ {
				weights[i][j][k] = 2*rand.Float64() - 1
			}
		}
	}

	return neuralNet{weights, biases, nodes, activ, wPartials, bPartials, cPartials}
}

func (nn *neuralNet) process(input []float64) []float64 {
	//fmt.Println("-----------------------------")
	//for i := 0; i < 32; i++ {
	//fmt.Println(input[i*32:i*32+32])
	//}
	for i := 0; i < len(nn.nodes[0]); i++ {
		nn.nodes[0][i] = input[i]
	}
	for i := 0; i < len(nn.Weights); i++ {
		for j := 0; j < len(nn.Weights[i]); j++ {
			activ := nn.Biases[i][j]
			for k := 0; k < len(nn.Weights[i][j]); k++ {
				activ += nn.nodes[i][k] * nn.Weights[i][j][k]
			}
			nn.activ[i][j] = activ
			nn.nodes[i+1][j] = 1 / (1 + math.Exp(-activ))
		}
	}
	return nn.nodes[len(nn.nodes)-1]
}

// TODO food for thought, this method is only called by train, and could sensibly go inside it, but then train would become thicc. Is it worth another method? It would actually make things a little simpler
func (nn *neuralNet) backprop(input, expected []float64) float64 {
	output := nn.process(input)
	cost := 0.0
	for i := 0; i < len(output); i++ {
		cost += (output[i] - expected[i]) * (output[i] - expected[i])
	}
	for i := len(nn.Weights) - 1; i >= 0; i-- {
		for j := 0; j < len(nn.Weights[i]); j++ {
			if i == len(nn.Weights)-1 {
				nn.cPartials[i][j] = 2 * (output[j] - expected[j])
			} else {
				nn.cPartials[i][j] = 0
				for k := 0; k < len(nn.Weights[i+1]); k++ {
					activ := nn.activ[i+1][k]
					nn.cPartials[i][j] += nn.Weights[i+1][k][j] * math.Exp(-activ) / ((1 + math.Exp(-activ)) * (1 + math.Exp(-activ))) * nn.cPartials[i+1][k]
				}
			}
			activ := nn.activ[i][j]
			activPartial := math.Exp(-activ) / ((1 + math.Exp(-activ)) * (1 + math.Exp(-activ)))
			for k := 0; k < len(nn.Weights[i][j]); k++ {
				nn.wPartials[i][j][k] = nn.nodes[i][k] * activPartial * nn.cPartials[i][j]
			}
			nn.bPartials[i][j] = activPartial * nn.cPartials[i][j]
		}
	}
	return cost
}

func (nn *neuralNet) train(input, expected [][]float64) {
	wPartialsTotal := make([][][]float64, len(nn.Weights))
	bPartialsTotal := make([][]float64, len(nn.Biases))
	for i := 0; i < len(nn.Weights); i++ {
		wPartialsTotal[i] = make([][]float64, len(nn.Weights[i]))
		bPartialsTotal[i] = make([]float64, len(nn.Biases[i]))
		for j := 0; j < len(nn.Weights[i]); j++ {
			wPartialsTotal[i][j] = make([]float64, len(nn.Weights[i][j]))
		}
	}
	costTotal := 0.0
	correctTotal := 0
	for i := 0; i < len(input); i++ {
		cost := nn.backprop(input[i], expected[i])
		costTotal += cost
		highest := 0
		for j := 1; j < len(nn.nodes[len(nn.nodes)-1]); j++ {
			if nn.nodes[len(nn.nodes)-1][j] > nn.nodes[len(nn.nodes)-1][highest] {
				highest = j
			}
		}
		//fmt.Println(nn.nodes[len(nn.nodes) - 1], expected[i], highest)
		if expected[i][highest] == 1 {
			correctTotal++
		}
		for j := 0; j < len(nn.Weights); j++ {
			for k := 0; k < len(nn.Weights[j]); k++ {
				bPartialsTotal[j][k] += nn.bPartials[j][k]
				for h := 0; h < len(nn.Weights[j][k]); h++ {
					wPartialsTotal[j][k][h] += nn.wPartials[j][k][h]
				}
			}
		}
	}
	fmt.Printf("Total cost of %d training examples: %f. Training accuracy: %f\n", len(input), costTotal, float64(correctTotal)/float64(len(input)))
	// TODO optimize learning rate
	prop := 0.005 / float64(len(input))
	for i := 0; i < len(nn.Weights); i++ {
		for j := 0; j < len(nn.Weights[i]); j++ {
			nn.Biases[i][j] -= prop * bPartialsTotal[i][j]
			for k := 0; k < len(nn.Weights[i][j]); k++ {
				nn.Weights[i][j][k] -= prop * wPartialsTotal[i][j][k]
			}
		}
	}
}

func main() {
	rand.Seed(time.Now().UnixNano())

	nn := newNeuralNet([]int{1024, 16, 16, 10})
	file, err := os.Open("nn.txt")
	if os.IsNotExist(err) {
		fmt.Println("Creating new random neural net")
	} else if err != nil {
		fmt.Println("error opening nn.txt:", err)
		return
	} else {
		dec := gob.NewDecoder(file)
		err := dec.Decode(&nn)
		if err != nil {
			fmt.Println("error decoding nn.txt", err)
			return
		}
		fmt.Println("decoded nn from nn.txt")
	}
	file.Close()
	//fmt.Println(nn.process(make([]float64, 1024)))
	//nn.backprop(make([]float64, 1024), []float64{0, 0, 0, 0, 1, 0, 0, 0, 0, 0})

	training := make([][]os.FileInfo, 10)
	for i := 0; i < 10; i++ {
		training[i], err = ioutil.ReadDir(fmt.Sprintf("out/train/%d", i))
		if err != nil {
			fmt.Println(err)
			return
		}
	}
	batchInputs := make([][]float64, 100)
	batchExpected := make([][]float64, 100)
	for batchNum := 0; true; batchNum++ {
		start := time.Now()
		for i := 0; i < 10; i++ {
			files := training[i]
			for j := 0; j < 10; j++ {
				idx := rand.Intn(len(files))
				file, err := os.Open(fmt.Sprintf("out/train/%d/%s", i, files[idx].Name()))
				if err != nil {
					fmt.Println(err)
					return
				}
				img, err := png.Decode(file)
				if err != nil {
					fmt.Println(err)
					return
				}
				file.Close()
				w, h := img.Bounds().Max.X, img.Bounds().Max.Y
				batchInputs[10*i+j] = make([]float64, w*h)
				for x := 0; x < w; x++ {
					for y := 0; y < h; y++ {
						r, _, _, _ := img.At(x, y).RGBA()
						batchInputs[10*i+j][y*w+x] = float64(r) / 65535
					}
				}
				batchExpected[10*i+j] = make([]float64, 10)
				batchExpected[10*i+j][i] = 1
			}
		}
		elapsed := time.Now().Sub(start)
		start = time.Now()
		nn.train(batchInputs, batchExpected)
		fmt.Printf("%s\t%s\n", elapsed, time.Now().Sub(start))
		if batchNum%500 == 0 {
			fmt.Println("Saving nn")
			file, err := os.Create("nn.txt")
			if err != nil {
				fmt.Println("err creating nn.txt:", err)
				return
			}
			enc := gob.NewEncoder(file)
			err = enc.Encode(nn)
			if err != nil {
				fmt.Println("err creating gob encoder:", err)
				return
			}
			file.Close()
		}
	}
}
