var myMap = L.map("map", {
    center: [40.7128, -74.0059],
    zoom: 11,
  })
  
  L.tileLayer("https://api.mapbox.com/styles/v1/{id}/tiles/{z}/{x}/{y}?access_token={accessToken}", {
    attribution: "© <a href='https://www.mapbox.com/about/maps/'>Mapbox</a> © <a href='http://www.openstreetmap.org/copyright'>OpenStreetMap</a> <strong><a href='https://www.mapbox.com/map-feedback/' target='_blank'>Improve this map</a></strong>",
    tileSize: 512,
    maxZoom: 18,
    zoomOffset: -1,
    id: "mapbox/streets-v11",
    accessToken: API_KEY
    }).addTo(myMap);
  
  var url = "data/lyft_sep5_2014.geojson";
  
  var lyft = new L.geoJson();
  lyft.addTo(myMap);
  
  $.ajax({
    dataType: "json",
    url: url,
    success: function(data) {
        $(data.features).each(function(key, data) {
            lyft.addData(data);
  
        });
      }
  })
  
 