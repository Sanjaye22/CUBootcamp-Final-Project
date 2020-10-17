function getData(value){
  if(value == "hour_data"){
    hourGraph()
  }
  else{
    groupGraph()
  }
};

groupGraph()

function groupGraph(){
  d3.select("#plot").html("");
  d3.csv("data/hour_grouping_count.csv").then(function(data1) {
    var hour_grouping = data1.map(x => x.hour_grouping);
    var grouping_counts = data1.map(x => x.hour_grouping_count);
    var data = [{
      type: "scatter",
      mode: "lines+markers",
      name: "Grouping Graph",
      x: hour_grouping,
      y: grouping_counts,
      line: {
        color: "#FF00BF"
      }
    }];
    
    var layout = {
      xaxis: {
        showgrid: false,
        showline: true,
        title: {
          text: 'Hour'
        }
    
      },
    
      yaxis: {
        showgrid: false,
        showline: true,
        title: {
          text: 'Pick Up Frequency'
          }
      }
    }
  Plotly.newPlot("plot", data, layout);
  })
}

function hourGraph(){
  d3.select("#plot").html("");
  d3.csv("data/hour_count.csv").then(function(data) {
    var hours = data.map(x => x.hour);
    var hour_counts = data.map(x => x.hour_count);
    
    data = [{
      type: "scatter",
      mode: "lines+markers",
      name: "Hour Graph",
      x: hours,
      y: hour_counts,
      line: {
        color: "#FF00BF"
      }
    }];

    var layout = {
      xaxis: {
        showgrid: false,
        showline: true,
        title: {
          text: 'Hour'
        }
    
      },
    
      yaxis: {
        showgrid: false,
        // showline: true,
        title: {
          text: 'Hour Counts'
          }
      }
    }

    Plotly.newPlot("plot", data, layout);
  
  })
}

