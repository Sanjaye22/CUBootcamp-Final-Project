// Creating a map object 
var myMap = L.map("map", {
  center: [40.7128, -74.0059],
  zoom: 11,
})

// Adding a tile layer
L.tileLayer("https://api.mapbox.com/styles/v1/{id}/tiles/{z}/{x}/{y}?access_token={accessToken}", {
  attribution: "© <a href='https://www.mapbox.com/about/maps/'>Mapbox</a> © <a href='http://www.openstreetmap.org/copyright'>OpenStreetMap</a> <strong><a href='https://www.mapbox.com/map-feedback/' target='_blank'>Improve this map</a></strong>",
  tileSize: 512,
  maxZoom: 18,
  zoomOffset: -1,
  id: "mapbox/streets-v11",
  accessToken: API_KEY
  }).addTo(myMap);

// Link to data 
var url = "data/lyft_sep5_2014.geojson";

// Grab data
d3.json(url, function(response) {

// Console log to see the data 
console.log(response);

// Create an array to hold the coordinates
var heatArray = [];

// Loop through the data 
var features = response.features
  for (var i = 0; i < features.length; i++) {
var location = features[i].geometry;

// Push data
if (location) {
heatArray.push([location.coordinates[1], location.coordinates[0]]);
  }
}

// Create heatlayer 
var heat = L.heatLayer(heatArray, {
    radius: 30,
    blur: 25
  }).addTo(myMap);
});

