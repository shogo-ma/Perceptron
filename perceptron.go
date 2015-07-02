package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

type Data struct {
	Words []string
	Label int
}

func signum(value int) int {
	if value >= 0 {
		return 1
	} else {
		return -1
	}
}

func extract_feature(file_name string) ([]Data, map[string]int) {
	file, err := os.Open(file_name)
	defer file.Close()
	if err != nil {
		panic(err)
	}

	scanner := bufio.NewScanner(file)

	// step1: feature extraction
	feature_dict := map[string]int{}
	datas := []Data{}
	for scanner.Scan() {
		line := strings.Fields(scanner.Text())
		for _, word := range line[1:] {
			if _, ok := feature_dict[word]; !ok {
				feature_dict[word] = len(feature_dict)
			}
		}
		label, err := strconv.Atoi(line[0])
		if err != nil {
			panic(err)
		}
		datas = append(datas, Data{line[1:], label})
	}
	return datas, feature_dict
}

func train(datas []Data, feature_dict map[string]int) map[int]int {
	weights := map[int]int{}
	for _, data := range datas {
		feature_vector := map[int]int{}
		for _, word := range data.Words {
			if _, ok := feature_dict[word]; ok {
				feature_vector[feature_dict[word]] += 1
			}
		}

		// calculate dot
		sum := 0
		for key, value := range feature_vector {
			sum += weights[key] * value
		}
		predict_label := signum(sum)
		if predict_label != data.Label {
			for key, value := range feature_vector {
				weights[key] += value * data.Label
			}
		}
	}
	return weights
}

func test(file_name string, weights map[int]int, feature_dict map[string]int) []int {
	file, err := os.Open(file_name)
	defer file.Close()
	if err != nil {
		panic(err)
	}

	labels := []int{}
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		words := strings.Fields(scanner.Text())
		feature_vector := map[int]int{}
		for _, word := range words {
			if _, ok := feature_dict[word]; ok {
				feature_vector[feature_dict[word]] += 1
			}
		}
		sum := 0
		for key, value := range feature_vector {
			sum += weights[key] * value
		}
		label := signum(sum)
		labels = append(labels, label)
	}
	return labels
}

func main() {

	// step1: extract feature
	datas, feature_dict := extract_feature("./data/titles-en-test.labeled")

	// step2: train
	weights := train(datas, feature_dict)

	// step3: test
	labeles := test("./data/titles-en-test.word", weights, feature_dict)
	fmt.Println(labeles)
}
