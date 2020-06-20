import React from "react";
import { Grid, Typography } from "@material-ui/core";

function SplashPage() {
  const titleStyles = {
    margin: 100,
  };
  return (
    <Grid container justify="center">
      <Grid item>
        <Typography variant="h3" align="center" style={titleStyles}>
          Find signal in your data with Apollo Analytics
        </Typography>
      </Grid>
    </Grid>
  );
}

export default SplashPage;
