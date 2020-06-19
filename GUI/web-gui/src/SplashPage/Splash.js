import React from "react";
import { Grid } from "@material-ui/core";
import SplashHeader from "./SplashHeader";
import SplashContent from "./SplashContent";
import SignInDialog from "./SignIn";
import RegisterDialog from "./Register";

class Splash extends React.Component {
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
  }

  handleRegister() {
    this.setState({ signInOpen: false, registerOpen: false });
  }

  render() {
    const bgstyles = {
      background:
        "linear-gradient(47deg,rgba(255,255,255,1) 0%, rgba(236,239,241,1) 100%)",
    };

    return (
      <div>
        <Grid container direction="column" style={bgstyles}>
          <Grid item>
            <SplashHeader clickHandler={this.handleClick} />
          </Grid>
          <Grid item container>
            <Grid item xs={false} sm={1} />
            <Grid item xs={12} sm={10}>
              <SplashContent />
            </Grid>
            <Grid item xs={false} sm={1} />
          </Grid>
        </Grid>
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

export default Splash;
