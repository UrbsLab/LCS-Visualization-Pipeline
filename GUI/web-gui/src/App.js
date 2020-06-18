import React from "react";
import { Grid } from "@material-ui/core";
import SplashHeader from "./SplashPage/SplashHeader";
import SplashContent from "./SplashPage/SplashContent";

class App extends React.Component {
  render() {
    const bgstyles = {
      background:
        "linear-gradient(47deg,rgba(255,255,255,1) 0%, rgba(236,239,241,1) 100%)",
    };

    return (
      <Grid container direction="column" style={bgstyles}>
        <Grid item>
          <SplashHeader />
        </Grid>
        <Grid item container>
          <Grid item xs={false} sm={1} />
          <Grid item xs={12} sm={10}>
            <SplashContent />
          </Grid>
          <Grid item xs={false} sm={1} />
        </Grid>
      </Grid>
    );
  }
}

export default App;
