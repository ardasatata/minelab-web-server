/**
 *
 * RecordVideo
 *
 */
import * as React from 'react';
import { useEffect, useState } from 'react';
import styled from 'styled-components/macro';
import { StyleConstants } from '../../../styles/StyleConstants';
import { PageWrapper } from '../../components/PageWrapper';
import { Helmet } from 'react-helmet-async';
import { io } from 'socket.io-client';
import {
  LoadingOutlined,
  PlayCircleOutlined,
  StopOutlined,
} from '@ant-design/icons';
import DataTable from 'react-data-table-component';

import {
  CategoryScale,
  Chart as ChartJS,
  Legend,
  LinearScale,
  LineElement,
  PointElement,
  Title,
  Tooltip,
  RadialLinearScale,
  ArcElement,
} from 'chart.js';
import { Line, PolarArea } from 'react-chartjs-2';
import { NavBar } from '../../components/NavBar';
import { useQuery } from '../../../hooks/useQuery';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  RadialLinearScale,
  ArcElement,
);

export const options = {
  responsive: true,
  plugins: {
    legend: {
      position: 'top' as const,
    },
    title: {
      display: true,
      text: 'Per Frame Stats',
    },
  },
  scales: {
    x: {
      display: true,
      title: {
        display: true,
      },
    },
    y: {
      display: true,
      title: {
        display: true,
        text: 'Value',
      },
      suggestedMin: -30,
      suggestedMax: 30,
    },
  },
  elements: {
    point: {
      radius: 0,
    },
  },
};

export const POLAR_DATA_EXAMPLE = {
  labels: [
    'Head Position',
    'Body Position',
    'Shoulder Position',
    'Right Arm',
    'Left Arm',
    'Bow Position',
  ],
  datasets: [
    {
      label: '# of Votes',
      data: [12, 10, 3, 5, 2, 3],
      backgroundColor: [
        'rgba(255, 99, 132, 0.5)',
        'rgba(54, 162, 235, 0.5)',
        'rgba(255, 206, 86, 0.5)',
        'rgba(75, 192, 192, 0.5)',
        'rgba(153, 102, 255, 0.5)',
        'rgba(255, 159, 64, 0.5)',
      ],
      borderWidth: 1,
    },
  ],
};

const customStyles = {
  rows: {
    style: {
      height: '24px', // override the row height
    },
  },
  headCells: {
    style: {
      paddingLeft: '8px', // override the cell padding for head cells
      paddingRight: '8px',
      fontWeight: 'bold',
    },
  },
};

const columns = [
  {
    name: 'Name',
    selector: row => row[2],
    style: {
      fontWeight: 'bold',
    },
  },
  {
    name: 'Value',
    selector: row => row[3],
  },
  {
    name: 'Error',
    selector: row => row[4],
    conditionalCellStyles: [
      {
        when: row => row[4] === 'Normal',
        style: {
          backgroundColor: 'rgba(63, 195, 128, 0.9)',
          color: 'white',
          '&:hover': {
            cursor: 'pointer',
          },
        },
      },
      {
        when: row => row[4] !== 'Normal',
        style: {
          backgroundColor: 'rgba(242, 38, 19, 0.9)',
          color: 'white',
          '&:hover': {
            cursor: 'not-allowed',
          },
        },
      },
      // You can also pass a callback to style for additional customization
      {
        when: row => row.calories < 600,
        style: row => ({ backgroundColor: row.isSpecial ? 'pink' : 'inerit' }),
      },
    ],
  },
];

interface Props {}

export function PlayVideo(props: Props) {
  const query = useQuery();
  // console.log(query.get('title'));
  // @ts-ignore
  const [socket, setSocket] = useState<any>(null);
  const [frame, setFrame] = useState<any>(0);
  const [img, setImg] = useState<string>('');
  const [data, setData] = useState<any>(null);

  const [labels, setLabels] = useState<Array<number>>([0]);

  const [chartData, setChartData] = useState<any>({
    labels: labels,
    datasets: [
      {
        label: 'Head',
        data: [0],
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.5)',
        cubicInterpolationMode: 'monotone',
        tension: 0.4,
      },
      {
        label: 'Body',
        data: [0],
        borderColor: 'rgb(53, 162, 235)',
        backgroundColor: 'rgba(54, 162, 235, 0.5)',
        cubicInterpolationMode: 'monotone',
        tension: 0.4,
      },
      {
        label: 'Shoulder',
        data: [0],
        borderColor: 'rgba(255,194,44,0.85)',
        backgroundColor: 'rgba(255, 206, 86, 0.5)',
        cubicInterpolationMode: 'monotone',
        tension: 0.4,
      },
      {
        label: 'Right Arm',
        data: [0],
        borderColor: 'rgb(75,192,192)',
        backgroundColor: 'rgba(75, 192, 192, 0.5)',
      },
      {
        label: 'Left Arm',
        data: [0],
        borderColor: 'rgb(153,102,255)',
        backgroundColor: 'rgba(153, 102, 255, 0.5)',
      },
      {
        label: 'Bow',
        data: [0],
        borderColor: 'rgb(255,159,64)',
        backgroundColor: 'rgba(255, 159, 64, 0.5)',
      },
    ],
  });

  const [polarData, setPolarData] = useState<any>({
    labels: [
      'Head Position',
      'Body Position',
      'Shoulder Position',
      'Right Arm',
      'Left Arm',
      'Bow Position',
    ],
    datasets: [
      {
        label: '# of Votes',
        // data: [12, 10, 3, 5, 2, 3],
        data: [0, 0, 0, 0, 0, 0],
        backgroundColor: [
          'rgba(255, 99, 132, 0.5)',
          'rgba(54, 162, 235, 0.5)',
          'rgba(255, 206, 86, 0.5)',
          'rgba(75, 192, 192, 0.5)',
          'rgba(153, 102, 255, 0.5)',
          'rgba(255, 159, 64, 0.5)',
        ],
        borderWidth: 1,
      },
    ],
  });

  // @ts-ignore
  useEffect(() => {
    const newSocket = io(`http://140.115.51.243:5000/work`, {
      transports: ['websocket'],
      upgrade: false,
    });
    // @ts-ignore
    newSocket.connect();
    setSocket(newSocket);

    newSocket.on('image', data => {
      const obj = JSON.parse(data);
      setFrame(obj.frame);
      setData(obj.data);
      setImg('data:image/jpeg;base64,' + obj.image);

      let labelsNew = labels;

      labelsNew.push(obj.frame);
      setLabels(Object.assign({}, labels, labelsNew));

      increment(obj);
    });
    return () => newSocket.close();
  }, [setSocket]);

  const play = () => {
    socket.emit('play', query.get('title'));
  };

  const stop = () => {
    socket.emit('stop');
    // eslint-disable-next-line no-restricted-globals
    location.reload();
  };

  const increment = obj => {
    let headValue = parseInt(obj.data[0][3]);
    let bodyValue = parseInt(obj.data[1][3]);
    let shoulderValue = parseInt(obj.data[2][3]);
    let rightArm = parseInt(obj.data[3][3]);
    let leftArm = parseInt(obj.data[4][3]);
    let bowPosition = parseInt(obj.data[5][3]);

    let datasetsCopy = chartData.datasets.slice(0);

    let copyHead = datasetsCopy[0].data;
    let copyBody = datasetsCopy[1].data;
    let copyShoulder = datasetsCopy[2].data;
    let copyRightArm = datasetsCopy[3].data;
    let copyLeftArm = datasetsCopy[4].data;
    let copyBow = datasetsCopy[5].data;

    copyHead.push(headValue);
    copyBody.push(bodyValue);
    copyShoulder.push(shoulderValue);
    copyRightArm.push(rightArm);
    copyLeftArm.push(leftArm);
    copyBow.push(bowPosition);

    let polarDataCopy = polarData.datasets.slice(0);

    let polarCopy = polarDataCopy[0].data;

    let headStats = obj.data[0][4];
    let bodyStats = obj.data[1][4];
    let shoulderStats = obj.data[2][4];
    let rightArmStats = obj.data[3][4];
    let leftArmStats = obj.data[4][4];
    let bowPositionStats = obj.data[5][4];

    if (headStats !== 'Normal') {
      polarCopy[0] = polarCopy[0] + 1;
    }

    if (bodyStats !== 'Normal') {
      polarCopy[1] = polarCopy[1] + 1;
    }

    if (shoulderStats !== 'Normal') {
      polarCopy[2] = polarCopy[2] + 1;
    }

    if (rightArmStats !== 'Normal') {
      polarCopy[3] = polarCopy[3] + 1;
    }

    if (leftArmStats !== 'Normal') {
      polarCopy[4] = polarCopy[4] + 1;
    }

    if (bowPositionStats !== 'Normal') {
      polarCopy[5] = polarCopy[5] + 1;
    }

    setChartData(
      Object.assign({}, chartData, {
        datasets: datasetsCopy,
      }),
    );

    setPolarData(
      Object.assign({}, polarData, {
        datasets: polarDataCopy,
      }),
    );
  };

  return (
    <>
      <Helmet>
        <title>Home Page</title>
        <meta
          name="description"
          content="A React Boilerplate application homepage"
        />
      </Helmet>
      <NavBar />
      <PageWrapperMain>
        <div className={'flex h-full w-3/5 bg-black'}>
          <div className={'flex flex-col w-full h-full justify-center'}>
            <div className={'flex flex-row'}>
              <div
                className={
                  'flex-1 text-white text-4xl py-4 transition ease-in-out bg-green-500 hover:bg-indigo-700 duration-300 items-center justify-center flex cursor-pointer'
                }
                onClick={play}
              >
                <PlayCircleOutlined className={'mr-2'} />
                {'PLAY'}
              </div>
              <div
                className={
                  'flex-1 text-white text-4xl py-4 transition ease-in-out bg-red-500 hover:bg-pink-700 duration-300 items-center justify-center flex cursor-pointer'
                }
                onClick={stop}
              >
                <StopOutlined className={'mr-2'} />
                {'STOP'}
              </div>
            </div>
            {socket ? (
              <div className="flex w-full flex-1">
                <div
                  className={
                    'text-white text-3xl absolute mt-4 ml-4 transition ease-in-out bg-black p-2'
                  }
                >
                  {`Frame Number : ${frame}`}
                </div>
                {data ? (
                  <img
                    src={img}
                    alt={'main-stream'}
                    className={'h-full w-full object-contain'}
                  />
                ) : (
                  <div className={'text-white text-9xl m-auto'} onClick={play}>
                    <LoadingOutlined />
                  </div>
                )}
              </div>
            ) : (
              <div>Not Connected</div>
            )}
          </div>
        </div>
        <div className={'h-full flex flex-col bg-white w-2/5 overflow-scroll'}>
          {data ? (
            <div className={'h-full mb-8'}>
              <DataTable
                columns={columns}
                data={data}
                customStyles={customStyles}
                dense={true}
              />
            </div>
          ) : null}
          <div className={'h-full mb-4'}>
            <Line options={options} data={chartData} />
          </div>
          <div className={'px-12 mb-24'}>
            <PolarArea
              data={polarData}
              options={{
                plugins: {
                  legend: {
                    position: 'top' as const,
                  },
                  title: {
                    display: true,
                    text: 'Error Plot',
                  },
                },
              }}
            />
          </div>
        </div>
      </PageWrapperMain>
    </>
  );
}

const Wrapper = styled.header`
  box-shadow: 0 1px 0 0 ${p => p.theme.borderLight};
  height: ${StyleConstants.NAV_BAR_HEIGHT};
  display: flex;
  position: fixed;
  top: 0;
  width: 100%;
  background-color: ${p => p.theme.background};
  z-index: 2;

  @supports (backdrop-filter: blur(10px)) {
    backdrop-filter: blur(10px);
    background-color: ${p =>
      p.theme.background.replace(
        /rgba?(\(\s*\d+\s*,\s*\d+\s*,\s*\d+)(?:\s*,.+?)?\)/,
        'rgba$1,0.75)',
      )};
  }

  ${PageWrapper} {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
`;

export const PageWrapperMain = styled.div`
  display: flex;
  margin: 0 auto;
  box-sizing: content-box;
  height: calc(100vh - ${StyleConstants.NAV_BAR_HEIGHT});
`;
