import React from "react";
import {
  Dialog,
  TextField,
  Button,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
} from "@material-ui/core";

class SignInDialog extends React.Component {
  render() {
    const passwordStyle = {
      color: "#bdbdbd",
      fontSize: 10,
    };
    return (
      <Dialog
        open={this.props.shouldDisplay}
        onClose={this.props.close}
        aria-labelledby="form-dialog-title"
      >
        <DialogTitle id="form-dialog-title">
          Sign in to your Apollo Dashboard
        </DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            id="username"
            label="email address"
            type="email"
            fullWidth
          />
          <TextField
            autoFocus
            margin="dense"
            id="password"
            label="password"
            type="password"
            fullWidth
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={this.props.close} color="primary">
            <p style={passwordStyle}>Forgot Password?</p>
          </Button>
          <Button onClick={this.props.signin} color="primary">
            Sign In
          </Button>
        </DialogActions>
      </Dialog>
    );
  }
}

export default SignInDialog;
