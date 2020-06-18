import React from "react";
import { AppBar, Toolbar, Typography, Button, Grid } from "@material-ui/core";

class Header extends React.Component {
  handleClick(event) {
    const { name } = event.currentTarget;
    if (name === "sign-in") {
      console.log("sign in");
    } else if (name === "register") {
      console.log("register");
    }
  }

  render() {
    const buttonStyle = {
      paddingRight: 15,
      paddingLeft: 15,
      paddingTop: 10,
      paddingBottom: 10,
    };

    return (
      <AppBar position="static" color="secondary">
        <Toolbar>
          <Grid container justify="flex-start">
            <Grid item>
              <Typography variant="h4">Theseus</Typography>
            </Grid>
          </Grid>
          <Grid container justify="flex-end" spacing={2}>
            <Grid item>
              <Button
                color="primary"
                variant="contained"
                style={buttonStyle}
                onClick={this.handleClick}
                name="sign-in"
              >
                Sign In
              </Button>
            </Grid>
            <Grid item>
              <Button
                color="primary"
                variant="contained"
                style={buttonStyle}
                onClick={this.handleClick}
                name="register"
              >
                Register
              </Button>
            </Grid>
          </Grid>
        </Toolbar>
      </AppBar>
    );
  }
}

export default Header;
