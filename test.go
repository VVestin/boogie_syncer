package main

import (
	"strconv"
	"fmt"
	"gocv.io/x/gocv"
	"boogie/learn"
	"image"
	"image/color"
	"os"
	"sort"
)

type symbol struct {
	cnt  []image.Point
	rect image.Rectangle
	img  gocv.Mat
	number string // lol
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func genIdNumbers(fileName string) {
	emptyMat := gocv.Mat{}

	_ = os.Mkdir(fmt.Sprintf("out/%s", fileName), os.ModePerm)

	img := gocv.IMRead(fmt.Sprintf("res/%s.png", fileName), gocv.IMReadColor)
	fmt.Printf("img (%d, %d)\n", img.Rows(), img.Cols())
	defer img.Close()
	grey := gocv.NewMat()
	defer grey.Close()
	thresh := gocv.NewMat()
	defer thresh.Close()
	blur := gocv.NewMat()
	defer blur.Close()

	gocv.CvtColor(img, &grey, gocv.ColorBGRToGray)
	gocv.GaussianBlur(grey, &blur, image.Pt(5, 5), 0, 0, gocv.BorderReplicate)
	gocv.Threshold(blur, &thresh, 0, 255, gocv.ThresholdBinaryInv+gocv.ThresholdOtsu)
	contours := gocv.FindContours(thresh, gocv.RetrievalList, gocv.ChainApproxSimple)

	var box image.Rectangle
	fmt.Println(box)
	for i, c := range contours {
		r := gocv.BoundingRect(c)
		if r.Max.X-r.Min.X < img.Cols()/10 || r.Max.X < img.Cols()-30 || r.Min.Y > 15 || r.Max.Y > img.Rows()/8 {
			continue
		}
		box = r
		//gocv.DrawContours(&img, contours, i, color.RGBA{uint8(rand.Int() % 255), uint8(rand.Int() % 255), uint8(rand.Int() % 255), 0}, 2)
		//gocv.Rectangle(&img, box, color.RGBA{0, 255, 0, 0}, 2)
		fmt.Printf("box: %d - %s\n", i, c)
	}
	emptyRect := image.Rectangle{}
	if box == emptyRect {
		fmt.Printf("error. boxed id number not detected for %s\n", fileName)
		return
	}

	symbols := make([]symbol, 0)
	for _, c := range contours {
		r := gocv.BoundingRect(c)
		if r.Min.X > box.Min.X && r.Max.Y < box.Max.Y {
			symbols = append(symbols, symbol{cnt: c, rect: r})
		}
	}
	sort.Slice(symbols, func(i, j int) bool {
		return symbols[i].rect.Min.X < symbols[j].rect.Min.X
	})

SymbolLoop:
	for i, s := range symbols {
		// Check if s is contained within another contour
		for j := i - 1; j >= 0; j-- {
			in, out := s.rect, symbols[j].rect
			if in.Max.X > out.Max.X || in.Min.Y < out.Min.Y || in.Max.Y > out.Max.Y || symbols[j].img == emptyMat {
				continue
			}
			//gocv.DrawContours(&img, [][]image.Point{s.cnt}, 0, color.RGBA{255, 0, 0, 255}, -1)
			// s in completely contained within symbols[j]
			adjCnt := make([]image.Point, len(s.cnt))
			for k, p := range s.cnt {
				adjCnt[k] = image.Pt(2*(p.X-out.Min.X+1), 2*(p.Y-out.Min.Y+1))
			}
			//fmt.Printf("%d: %v\nis within\n%d: %v\n", i, s, j, symbols[j])
			gocv.DrawContours(&symbols[j].img, [][]image.Point{adjCnt}, 0, color.RGBA{0, 0, 0, 255}, -1)
			continue SymbolLoop
		}

		//fmt.Println(s.cnt, (s.rect.Max.X - s.rect.Min.X) * (s.rect.Max.Y - s.rect.Min.Y))
		if (s.rect.Max.X - s.rect.Min.X) * (s.rect.Max.Y - s.rect.Min.Y) < 30 {
			symbols[i].number = "-"
			continue
		}

		//gocv.DrawContours(&img, [][]image.Point{s.cnt}, 0, color.RGBA{0, 255, 0, 255}, -1)
		adjCnt := make([]image.Point, len(s.cnt))
		for j, p := range s.cnt {
			adjCnt[j] = image.Pt(2*(p.X-s.rect.Min.X+1), 2*(p.Y-s.rect.Min.Y+1))
		}
		r := gocv.BoundingRect(adjCnt)
		s.img = gocv.NewMatWithSize(r.Max.Y-r.Min.Y+4, r.Max.X-r.Min.X+4, gocv.MatTypeCV8U)
		gocv.Rectangle(&s.img, image.Rect(0, 0, s.img.Cols(), s.img.Rows()), color.RGBA{0, 0, 0, 255}, -1)
		gocv.DrawContours(&s.img, [][]image.Point{adjCnt}, 0, color.RGBA{255, 255, 255, 255}, -1)
		symbols[i] = s
	}

	nn := learn.NewNeuralNet([]int{1024, 64, 10})
	nn.Load("nn.txt")

	docID := ""
	for i, s := range symbols {
		if s.img == emptyMat { // || is redundant, but nice
			docID += s.number
			continue
		}
		resized := gocv.NewMat()
		gocv.Resize(s.img, &resized, image.Point{min(32, max(16, int(32 * float64(s.img.Cols()) / float64(s.img.Rows())))), 32}, 0, 0, gocv.InterpolationCubic)

		//fmt.Printf("%d: (%d, %d) -> (%d, %d)\n", i, s.img.Cols(), s.img.Rows(), resized.Cols(), resized.Rows())
		imgPath := fmt.Sprintf("out/%s/%d.png", fileName, i)
		gocv.IMWrite(imgPath, resized)
		_, number := nn.Process(learn.GetImageData(imgPath))
		symbols[i].number = strconv.Itoa(number)
		docID += symbols[i].number

		s.img.Close()
		resized.Close()
	}
	fmt.Println("document number:", docID)

	gocv.IMWrite(fmt.Sprintf("out/%s/out1.png", fileName), thresh)
	//gocv.IMWrite(fmt.Sprintf("out/%s/out2.png", fileName), blur)
	//gocv.IMWrite(fmt.Sprintf("out/%s/out3.png", fileName), grey)
	//gocv.IMWrite(fmt.Sprintf("out/%s/out4.png", fileName), img)
	gocv.IMWrite(fmt.Sprintf("out/%s.png", docID), img)
}

func main() {
	if len(os.Args) != 2 {
		fmt.Println("error. expected file name as argument")
		return
	}
	genIdNumbers(os.Args[1])
}
