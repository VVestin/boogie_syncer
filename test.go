package main

import (
	"fmt"
	"gocv.io/x/gocv"
	"image"
	"image/color"
	"math/rand"
	"sort"
)

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
	img := gocv.IMRead(fileName, gocv.IMReadColor)
	defer img.Close()
	fmt.Printf("img (%d, %d) type: %d\n", img.Cols(), img.Rows(), img.Type())
	grey := gocv.NewMat()
	defer grey.Close()
	thresh := gocv.NewMat()
	defer thresh.Close()
	blur := gocv.NewMat()
	defer blur.Close()

	gocv.CvtColor(img, &grey, gocv.ColorBGRToGray)
	gocv.GaussianBlur(grey, &blur, image.Pt(5, 5), 0, 0, gocv.BorderReplicate)
	gocv.Threshold(blur, &thresh, 0, 255, gocv.ThresholdBinaryInv+gocv.ThresholdOtsu)

	contours := gocv.FindContours(thresh, gocv.RetrievalTree, gocv.ChainApproxSimple)
	var box image.Rectangle
	for i, c := range contours {
		r := gocv.BoundingRect(c)
		if r.Max.X-r.Min.X < img.Cols()/10 || r.Max.X < img.Cols()-30 || r.Min.Y > 15 || r.Max.Y > img.Rows()/8 {
			continue
		}
		box = r
		gocv.DrawContours(&img, contours, i, color.RGBA{uint8(rand.Int() % 255), uint8(rand.Int() % 255), uint8(rand.Int() % 255), 0}, 2)
		gocv.Rectangle(&img, box, color.RGBA{0, 255, 0, 0}, 2)
		fmt.Printf("box: %d - %s\n", i, c)
	}

	symbols := make([][]image.Point, 0)
	for _, c := range contours {
		r := gocv.BoundingRect(c)
		if r.Min.X > box.Min.X && r.Max.Y < box.Max.Y {
			symbols = append(symbols, c)
		}
	}
	sort.Slice(symbols, func(i, j int) bool {
		ri, rj := gocv.BoundingRect(symbols[i]), gocv.BoundingRect(symbols[j])
		return ri.Min.X < rj.Min.X
	})

	for i, c := range symbols {
		gocv.DrawContours(&img, symbols, i, color.RGBA{uint8(rand.Int() % 255), uint8(rand.Int() % 255), uint8(rand.Int() % 255), 0}, 2)
		r := gocv.BoundingRect(c)
		for i, p := range c {
			c[i] = image.Point{p.X - r.Min.X + 2, p.Y - r.Min.Y + 2}
		}
		fmt.Printf("symbol %d: %s\n", i, symbols[i])
		symbolImg := gocv.NewMatWithSize(max(16, r.Max.Y-r.Min.Y+4), max(16, r.Max.X-r.Min.X+4), gocv.MatTypeCV8U)
		gocv.Rectangle(&symbolImg, image.Rect(0, 0, symbolImg.Cols(), symbolImg.Rows()), color.RGBA{0, 0, 0, 255}, -1)
		gocv.DrawContours(&symbolImg, symbols, i, color.RGBA{255, 255, 255, 255}, -1)

		symbolImgResized := gocv.NewMat()
		gocv.Resize(symbolImg, &symbolImgResized, image.Point{32, 32}, 0, 0, gocv.InterpolationLinear)

		gocv.IMWrite(fmt.Sprintf("out/zym%d.png", i), symbolImg)
		gocv.IMWrite(fmt.Sprintf("out/sym%d.png", i), symbolImgResized)

		symbolImg.Close()
		symbolImgResized.Close()
	}

	gocv.IMWrite("out1.png", thresh)
	gocv.IMWrite("out2.png", blur)
	gocv.IMWrite("out3.png", grey)
	gocv.IMWrite("out4.png", img)
	gocv.IMWrite(fmt.Sprintf("out/sym%d.png", len(symbols)), grey)
}

func main() {
	fmt.Println("Doing opencv stuff")
	genIdNumbers("im1.png")
}
