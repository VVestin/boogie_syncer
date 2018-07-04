package main

import (
	"fmt"
	"gocv.io/x/gocv"
	"image"
	"image/color"
	"sort"
	"os"
)

type symbol struct {
	cnt  []image.Point
	rect image.Rectangle
	img  gocv.Mat
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

func genTraining(fileName string) {
	emptyMat := gocv.Mat{}

	img := gocv.IMRead(fmt.Sprintf("res/train_raw/%s.png", fileName), gocv.IMReadColor)
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
	symbols := make([]symbol, len(contours))
	for i, c := range contours {
		symbols[i] = symbol{cnt: c, rect: gocv.BoundingRect(c)}
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
			gocv.DrawContours(&img, [][]image.Point{s.cnt}, 0, color.RGBA{255, 0, 0, 255}, -1)
			// s in completely contained within symbols[j]
			adjCnt := make([]image.Point, len(s.cnt))
			for k, p := range s.cnt {
				adjCnt[k] = image.Pt(2*(p.X-out.Min.X+1), 2*(p.Y-out.Min.Y+1))
			}
			//fmt.Printf("%d: %v\nis within\n%d: %v\n", i, s, j, symbols[j])
			gocv.DrawContours(&symbols[j].img, [][]image.Point{adjCnt}, 0, color.RGBA{0, 0, 0, 255}, -1)
			continue SymbolLoop
		}

		gocv.DrawContours(&img, [][]image.Point{s.cnt}, 0, color.RGBA{0, 255, 0, 255}, -1)
		adjCnt := make([]image.Point, len(s.cnt))
		for j, p := range s.cnt {
			adjCnt[j] = image.Pt(2*(p.X-s.rect.Min.X+1), 2*(p.Y-s.rect.Min.Y+1))
		}
		r := gocv.BoundingRect(adjCnt)
		s.img = gocv.NewMatWithSize(max(16, r.Max.Y-r.Min.Y+4), max(16, r.Max.X-r.Min.X+4), gocv.MatTypeCV8U)
		gocv.Rectangle(&s.img, image.Rect(0, 0, s.img.Cols(), s.img.Rows()), color.RGBA{0, 0, 0, 255}, -1)
		gocv.DrawContours(&s.img, [][]image.Point{adjCnt}, 0, color.RGBA{255, 255, 255, 255}, -1)
		symbols[i] = s
	}

	fileNum := 0
	symNum := 0
	for _, s := range symbols {
		if s.img == emptyMat {
			continue
		}
		for err := error(nil); err == nil || !os.IsNotExist(err); {
			fileNum++
			_, err = os.Stat(fmt.Sprintf("out/train/%s/%d.png", fileName, fileNum))
		}
		symNum++
		resized := gocv.NewMat()
		gocv.Resize(s.img, &resized, image.Point{32, 32}, 0, 0, gocv.InterpolationCubic)

		gocv.IMWrite(fmt.Sprintf("out/train/%s/%d.png", fileName, fileNum), resized)

		s.img.Close()
		resized.Close()
	}

	fmt.Printf("Generated %d training images from /res/train/%s.png\n", symNum, fileName)

	//gocv.IMWrite(fmt.Sprintf("out/train/%s/out1.png", fileName), thresh)
	//gocv.IMWrite(fmt.Sprintf("out/train/%s/out2.png", fileName), blur)
	//gocv.IMWrite(fmt.Sprintf("out/train/%s/out3.png", fileName), grey)
	//gocv.IMWrite(fmt.Sprintf("out/train/%s/out4.png", fileName), img)
}

func main() {
	fmt.Println("Doing opencv stuff")
	for i := 0; i < 10; i++ {
		genTraining(fmt.Sprintf("%d", i))
	}
}
