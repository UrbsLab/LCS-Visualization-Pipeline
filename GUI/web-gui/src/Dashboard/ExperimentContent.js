import React from "react";
import {
  Typography,
  Slider,
  Dialog,
  DialogContent,
  Button,
  ButtonGroup,
} from "@material-ui/core";

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
    this.clickTopToggle = this.clickTopToggle.bind(this);
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

  clickTopToggle(event) {
    const name = event.currentTarget.getAttribute("name");
    this.setState({ screen: name });
  }

  componentDidUpdate(prevProps, prevState) {
    if (
      prevProps.name !== this.props.name ||
      prevState.screen !== this.state.screen
    ) {
      this.setState({ clusterCount: 1 });
    }
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
    } else {
      imgsrc =
        "FakeFiles/" +
        this.props.name +
        "/Full/visualizations/rulepop/ruleclusters/" +
        this.state.clusterCount.toString() +
        "_clusters/ruleclustermap.png";
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

        <div style={{ textAlign: "center", marginBottom: 20 }}>
          <ButtonGroup color="primary">
            <Button
              variant={this.state.screen === "at" ? "contained" : "outlined"}
              name="at"
              onClick={this.clickTopToggle}
            >
              Attribute Tracking
            </Button>
            <Button
              variant={this.state.screen === "at" ? "outlined" : "contained"}
              name="rule"
              onClick={this.clickTopToggle}
            >
              Rule Population
            </Button>
          </ButtonGroup>
        </div>

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

        <div align="center" style={{ marginTop: 30 }}>
          {this.state.screen === "at" ? (
            <Typography variant="subtitle1">
              Number of Attribute Tracking Clusters
            </Typography>
          ) : (
            <Typography variant="subtitle1">
              Number of Rule Population Clusters
            </Typography>
          )}
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

        <div style={{ margin: "auto", textAlign: "center" }}>
          <Button color="primary" variant="outlined" style={{ margin: 10 }}>
            Download Cluster Analysis
          </Button>
          {this.state.screen === "at" ? (
            <Button color="primary" variant="outlined" style={{ margin: 10 }}>
              Download Cluster-Labelled Dataset
            </Button>
          ) : (
            <Button color="primary" variant="outlined" style={{ margin: 10 }}>
              Download Rule Co-Specificity Network
            </Button>
          )}
        </div>

        <Dialog
          open={this.state.imageExpanded}
          onClose={this.imageClickOff}
          fullWidth={true}
          maxWidth={"xl"}
        >
          <DialogContent>
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
