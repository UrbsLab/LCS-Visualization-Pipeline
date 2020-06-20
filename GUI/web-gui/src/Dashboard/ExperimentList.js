import React from "react";
import {
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
  Typography,
  CircularProgress,
} from "@material-ui/core";
import { withStyles } from "@material-ui/core/styles";
import { LibraryBooksRounded } from "@material-ui/icons";

const styles = (theme) => ({
  root: {
    maxHeight: "100%",
    maxWidth: "100%",
    backgroundColor: theme.palette.secondary.light,
    overflow: "auto",
  },
  selected: {
    backgroundColor: theme.palette.primary.light,
  },
  deselected: {
    backgroundColor: theme.palette.secondary.light,
  },
});

class ExperimentList extends React.Component {
  render() {
    return (
      <List component="nav">
        {this.props.data.map((tile) => (
          <div
            onClick={this.props.handleClick}
            key={tile.experimentName}
            className={
              this.props.selectedExperiment === tile.experimentName
                ? this.props.classes.selected
                : this.props.classes.deselected
            }
            name={(tile.trained ? "t" : "f") + tile.experimentName}
          >
            <ExperimentTile
              name={tile.experimentName}
              created={tile.created}
              trained={tile.trained}
              selected={this.props.selectedExperiment === tile.experimentName}
            />
            <Divider />
          </div>
        ))}
      </List>
    );
  }
}

export default withStyles(styles)(ExperimentList);

function timeConverter(UNIX_timestamp) {
  var a = new Date(UNIX_timestamp * 1000);
  var months = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
  ];
  var year = a.getFullYear();
  var month = months[a.getMonth()];
  var date = a.getDate();
  var hour = a.getHours();
  var min = a.getMinutes();
  var sec = a.getSeconds();
  let time;
  if (date.toString().length === 2) {
    time = date + " " + month + " " + year;
  } else {
    time = "0" + date + " " + month + " " + year;
  }

  return time;
}

function ExperimentTile(props) {
  const colorPick = props.selected ? "white" : "black";
  const subheading =
    (props.trained ? "Trained" : "Still Training... ") +
    " | Created " +
    timeConverter(props.created);
  return (
    <ListItem
      button
      style={{
        paddingTop: 8,
        paddingBottom: 8,
        paddingLeft: 20,
      }}
    >
      <ListItemIcon>
        {props.trained ? (
          <LibraryBooksRounded style={{ color: colorPick }} />
        ) : (
          <CircularProgress
            style={{ color: colorPick, width: "50%", height: "50%" }}
          />
        )}
      </ListItemIcon>
      <ListItemText
        primary={
          <Typography
            variant="subtitle1"
            style={{ color: colorPick, margin: 0, padding: 0 }}
          >
            {props.name}
          </Typography>
        }
        secondary={
          <Typography
            variant="caption"
            style={{ color: colorPick, margin: 0, padding: 0 }}
          >
            {subheading}
          </Typography>
        }
      />
    </ListItem>
  );
}
