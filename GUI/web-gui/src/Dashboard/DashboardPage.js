import React from "react";
import { Grid, Button, Typography } from "@material-ui/core";
import DashboardHeader from "./DashboardHeader";
import userdata from "./FakeData/data";
import ExperimentList from "./ExperimentList";
import ExperimentDialog from "./ExperimentDialog";
import ExperimentContent from "./ExperimentContent";

class DashboardPage extends React.Component {
  constructor() {
    super();
    this.state = {
      userdata: null,
      isLoading: true,
      selectedExperiment: null,
      showAddPopup: false,
    };
    this.handleExperimentClick = this.handleExperimentClick.bind(this);
    this.handleAddExperimentButtonClick = this.handleAddExperimentButtonClick.bind(
      this
    );
    this.handleCloseAddExperiment = this.handleCloseAddExperiment.bind(this);
    this.handleAddExperiment = this.handleAddExperiment.bind(this);
  }

  componentDidMount() {
    this.setState({ userdata: userdata, isLoading: false });
  }

  handleExperimentClick(event) {
    const name = event.currentTarget.getAttribute("name");
    this.setState({ selectedExperiment: name });
  }

  handleAddExperimentButtonClick(event) {
    this.setState({ showAddPopup: true });
  }

  handleCloseAddExperiment() {
    this.setState({ showAddPopup: false });
  }

  handleAddExperiment() {
    this.setState({ showAddPopup: false });
  }

  render() {
    return (
      <div style={{ display: "block", maxHeight: "100vh" }}>
        <Grid container direction="column" style={{ display: "block" }}>
          <Grid item style={{ display: "block" }}>
            {this.state.isLoading ? (
              <DashboardHeader
                firstname=""
                lastname=""
                logout={this.props.logoutfunc}
              />
            ) : (
              <DashboardHeader
                firstname={this.state.userdata.firstname}
                lastname={this.state.userdata.lastname}
                logout={this.props.logoutfunc}
              />
            )}
          </Grid>
          <Grid item container>
            <Grid item xs={3}>
              <Grid container direction="column">
                <Grid item style={{ height: "88vh", overflow: "auto" }}>
                  {this.state.isLoading ? (
                    <h1>Loading...</h1>
                  ) : (
                    <ExperimentList
                      data={this.state.userdata.resultdata}
                      selectedExperiment={this.state.selectedExperiment}
                      handleClick={this.handleExperimentClick}
                    />
                  )}
                </Grid>
                <Grid item>
                  <Button
                    fullWidth
                    color="primary"
                    variant="contained"
                    style={{ height: "5vh" }}
                    onClick={this.handleAddExperimentButtonClick}
                  >
                    Add Experiment
                  </Button>
                </Grid>
              </Grid>
            </Grid>
            <Grid item xs={9}>
              <ExperimentContent name={this.state.selectedExperiment} />
            </Grid>
          </Grid>
        </Grid>
        <ExperimentDialog
          shouldDisplay={this.state.showAddPopup}
          close={this.handleCloseAddExperiment}
          addExperiment={this.handleAddExperiment}
        />
      </div>
    );
  }
}

export default DashboardPage;
