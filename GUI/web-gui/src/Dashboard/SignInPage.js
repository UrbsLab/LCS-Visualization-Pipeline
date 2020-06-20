import React from "react";
import Dashboard from "./Dashboard";
import { Box, Typography, Paper, Button } from "@material-ui/core";
import SignInDialog from "../SplashPage/SignIn";
import RegisterDialog from "../SplashPage/Register";

class SignInPage extends React.Component {
  constructor() {
    super();
    this.state = {
      signInOpen: false,
      registerOpen: false,
    };
    this.handleClick = this.handleClick.bind(this);
    this.handleClose = this.handleClose.bind(this);
    this.handleSignin = this.handleSignin.bind(this);
    this.handleRegister = this.handleRegister.bind(this);
  }

  handleClick(event) {
    const { name } = event.currentTarget;
    if (name === "sign-in") {
      console.log("sign in");
      if (!this.state.signInOpen) {
        this.setState({ signInOpen: true, registerOpen: false });
      }
    } else if (name === "register") {
      console.log("register");
      if (!this.state.registerOpen) {
        this.setState({ signInOpen: false, registerOpen: true });
      }
    }
  }

  handleClose() {
    this.setState({ signInOpen: false, registerOpen: false });
  }

  handleSignin() {
    this.setState({ signInOpen: false, registerOpen: false });
    this.props.loginfunc();
  }

  handleRegister() {
    this.setState({ signInOpen: false, registerOpen: false });
  }

  render() {
    return (
      <div>
        <Box
          display="flex"
          width={"100vw"}
          height={"100vh"}
          style={{
            background:
              "linear-gradient(47deg,rgba(255,255,255,1) 0%, rgba(236,239,241,1) 100%)",
          }}
        >
          <Box m="auto">
            <Paper style={{ padding: 40 }}>
              <Typography
                variant="h5"
                align="center"
                style={{ marginBottom: 20 }}
              >
                Apollo Dashboard
              </Typography>
              <Button
                variant="outlined"
                color="primary"
                style={{ margin: 5 }}
                name="sign-in"
                onClick={this.handleClick}
              >
                Sign In
              </Button>
              <Button
                variant="outlined"
                color="primary"
                style={{ margin: 5 }}
                name="register"
                onClick={this.handleClick}
              >
                Register
              </Button>
            </Paper>
          </Box>
        </Box>
        <SignInDialog
          shouldDisplay={this.state.signInOpen}
          close={this.handleClose}
          signin={this.handleSignin}
        />
        <RegisterDialog
          shouldDisplay={this.state.registerOpen}
          close={this.handleClose}
          register={this.handleRegister}
        />
      </div>
    );
  }
}

export default SignInPage;
