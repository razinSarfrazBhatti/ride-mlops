package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

var mlServiceURL = "http://ml-api:8000/predict"

type TripRequest struct {
	TripDistance   float64 `json:"trip_distance"`
	TripDuration   float64 `json:"trip_duration"`
	PickupHour     int     `json:"pickup_hour"`
	PassengerCount float64 `json:"passenger_count"`
}

type FareResponse struct {
	PredictedFare float64 `json:"predicted_fare"`
	Currency      string  `json:"currency"`
}

func predictFare(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var request TripRequest

	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, "Invalid JSON request", http.StatusBadRequest)
		return
	}

	if request.TripDistance <= 0 ||
		request.TripDuration <= 0 ||
		request.PickupHour < 0 ||
		request.PickupHour > 23 ||
		request.PassengerCount < 1 ||
		request.PassengerCount > 8 {
		http.Error(w, "Invalid trip details", http.StatusBadRequest)
		return
	}

	payload, err := json.Marshal(request)
	if err != nil {
		http.Error(w, "Failed to encode request", http.StatusInternalServerError)
		return
	}

	client := &http.Client{
		Timeout: 5 * time.Second,
	}

	response, err := client.Post(
		mlServiceURL,
		"application/json",
		bytes.NewBuffer(payload),
	)
	if err != nil {
		http.Error(w, "ML prediction service unavailable", http.StatusBadGateway)
		return
	}
	defer response.Body.Close()

	if response.StatusCode != http.StatusOK {
		http.Error(w, "ML prediction failed", http.StatusBadGateway)
		return
	}

	var result FareResponse

	if err := json.NewDecoder(response.Body).Decode(&result); err != nil {
		http.Error(w, "Invalid response from ML service", http.StatusBadGateway)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)
}

func main() {
	http.HandleFunc("/predict", predictFare)

	fmt.Println("Go backend listening on :8081")

	if err := http.ListenAndServe(":8081", nil); err != nil {
		fmt.Println("Server error:", err)
	}
}

// Keep io imported for the next exercise.
var _ io.Reader
