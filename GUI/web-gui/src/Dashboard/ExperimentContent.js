import React from "react";
import { Typography, Slider, Dialog, DialogContent } from "@material-ui/core";

class ExperimentContent extends React.Component {
  constructor() {
    super();
    this.state = {
      screen: "at", //at or rule
      clusterCount: 1,
      imageExpanded: false,
    };
    this.handleSlider = this.handleSlider.bind(this);
    this.imageClick = this.imageClick.bind(this);
    this.imageClickOff = this.imageClickOff.bind(this);
  }

  handleSlider(event, newValue) {
    this.setState({ clusterCount: newValue });
  }

  imageClick(event) {
    this.setState({ imageExpanded: true });
  }

  imageClickOff(event) {
    this.setState({ imageExpanded: false });
  }

  render() {
    let imgsrc;
    if (this.state.screen === "at") {
      imgsrc =
        "FakeFiles/" +
        this.props.name +
        "/Full/visualizations/at/atclusters/" +
        this.state.clusterCount.toString() +
        "_clusters/ATclustermap.png";
    }

    let clusterNum;
    for (var i = 0; i < this.props.data.length; i++) {
      if (this.props.data[i]["experimentName"] === this.props.name) {
        if (this.state.screen === "at") {
          clusterNum = this.props.data[i]["numATClusters"];
        } else {
          clusterNum = this.props.data[i]["numRuleClusters"];
        }
      }
    }

    return (
      <div>
        <Typography variant="h5" align="center" style={{ margin: 20 }}>
          {this.props.name}
        </Typography>

        <img
          src={imgsrc}
          alt="Failed to Load"
          style={{
            width: "40%",
            height: "40%",
            display: "block",
            marginLeft: "auto",
            marginRight: "auto",
          }}
          onClick={this.imageClick}
        />

        <div align="center">
          <Typography variant="subtitle1">Number of Clusters</Typography>
          <Slider
            value={this.state.clusterCount}
            defaultValue={1}
            step={1}
            min={1}
            max={clusterNum}
            valueLabelDisplay="auto"
            marks
            style={{ width: "50%" }}
            onChange={this.handleSlider}
          />
        </div>

        <Dialog open={this.state.imageExpanded} onClose={this.imageClickOff}>
          <DialogContent style={{ width: "80vh", height: "80vh" }}>
            <img
              src={imgsrc}
              alt="Failed to Load"
              style={{
                width: "80vh",
                height: "80vh",
                display: "block",
                marginLeft: "auto",
                marginRight: "auto",
              }}
            />
          </DialogContent>
        </Dialog>
      </div>
    );
  }
}

export default ExperimentContent;
