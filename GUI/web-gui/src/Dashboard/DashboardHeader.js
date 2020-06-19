import React from "react";
import { AppBar, Toolbar, Typography, Button, Grid } from "@material-ui/core";

class DashboardHeader extends React.Component {
  render() {
    const buttonStyle = {
      paddingRight: 15,
      paddingLeft: 15,
      paddingTop: 10,
      paddingBottom: 10,
    };
    const date = new Date();
    const hours = date.getHours();

    let timeOfDay;
    if (hours < 12 && hours >= 3) {
      timeOfDay = "Good Morning";
    } else if (hours >= 12 && hours < 17) {
      timeOfDay = "Good Afternoon";
    } else if (hours >= 17 && hours < 24) {
      timeOfDay = "Good Evening";
    } else {
      timeOfDay = "It's Getting Late ";
    }
    return (
      <AppBar position="static" color="secondary" style={{ height: "7vh" }}>
        <Toolbar style={{ height: "10vh" }}>
          <Grid container justify="flex-start">
            <Grid item>
              <Typography variant="h6">
                {timeOfDay} {this.props.firstname}
              </Typography>
            </Grid>
          </Grid>

          <Grid container justify="flex-end">
            <Grid item>
              <Button
                color="primary"
                variant="contained"
                style={buttonStyle}
                onClick={this.props.logout}
              >
                Sign Out
              </Button>
            </Grid>
          </Grid>
        </Toolbar>
      </AppBar>
    );
  }
}

export default DashboardHeader;
